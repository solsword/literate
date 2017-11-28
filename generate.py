"""
generate.py

Code for generating text from models.
"""

NUL = '\u0000'
STX = '\u0002'
ETX = '\u0003'

import dep

@dep.iter_task(("params", "texts"), "seed-{iter}")
def get_seed(iteration, params, texts):
  """
  Gets a seed fragment for text generation. Just pulls a random window_size
  fragment from anywhere in a random text.
  """
  random.seed(hash(iteration))
  text = random.choice(texts)
  text = STX * params["window_size"] + text + ETX * params["window_size"]
  start_index = random.randint(0, len(text) - params["window_size"]*2)

  return text[start_index:start_index + params["window_size"]]

dep.add_gather(
  ["lstm-generated-{}".format(i) for i in range(5)],
  "lstm-generated"
)

@dep.iter_task(("params","lstm-model","seed-{iter}"), "lstm-generated-{iter}")
def lstm_generate(params, model, seed=''):
  """
  Generates some text using the given model starting from the given seed. N
  specifies the number of additional characters to generate. If no seed is
  given, an STX block is used. The seed must be at least as long as the window
  size, or it will be pre-padded with STX characters.
  """
  if len(seed) < params["window_size"]:
    seed = STX * (params["window_size"] - len(seed)) + seed
  result = ''
  vec = vectorize(params, seed, pad=False)
  for i in range(params["generate_size"]):
    iv = vec[-params["window_size"]:]
    pred = model.predict(np.reshape(iv, (1,) + iv.shape), verbose=0)[0]
    nc = unichr(pred)
    result += nc
    nv = vectorize(params, nc, pad=False)
    vec = np.append(vec, nv, axis=0)

  return result


dep.add_gather(
  ["cnn-generated-{}".format(i) for i in range(5)],
  "cnn-generated"
)

@dep.iter_task(("params", "cnn-model", "seed-{iter}"), "cnn-generated-{iter}")
def cnn_generate(params, model, seed='', n=80):
  """
  Generates some text using the given model starting from the given seed. N
  specifies the number of additional characters to generate. If no seed is
  given, an STX block is used. The seed must be at least as long as the window
  size, or it will be pre-padded with STX characters.
  """
  if len(seed) < params["window_size"]:
    seed = STX * (params["window_size"] - len(seed)) + seed
  result = ''
  vec = vectorize(params, seed, pad=False)
  sv = vectorize(params, NUL, pad=False)
  for i in range(n):
    iv = vec[-(params["window_size"]-1):]
    iv = np.append(iv, sv, axis=0)
    pred = model.predict(np.reshape(iv, (1,) + iv.shape), verbose=0)[0]
    nc = unichr(pred[-1])
    result += nc
    nv = vectorize(params, nc, pad=False)
    vec = np.append(vec, nv, axis=0)

  return result

dep.add_gather(
  ["cnn-itergend-{}".format(i) for i in range(5)],
  "cnn-itergend"
)

@dep.iter_task(("params", "cnn-model"), "cnn-itergend-{iter}")
def cnn_itergen(params, model, limit=100):
  """
  Iteratively generates text by starting with random characters and
  autoencoding until convergence (or until the iteration limit is hit).
  """
  seed = ''.join(
    random.choice("abcdefghijklmnopqrstuvwxyz., \n")
      for x in range(params["window_size"])
  )
  print("SEED:", seed)
  last = seed
  vec = vectorize(params, seed, pad=False)
  for i in range(limit):
    pred = model.predict(np.reshape(vec, (1,) + vec.shape), verbose=0)[0]
    result = de_vectorize(params, pred)
    print("R", result)
    if result == last:
      break
    last = result
    vec = vectorize(params, result, pad=False)
  return result
