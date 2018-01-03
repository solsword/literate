"""
rate.py

Code for computing ratings from a model.
"""

import numpy as np

import dep
import utils
import vectorize

STX = '\u0002'
ETX = '\u0003'

def cat_crossent(true, pred):
  """
  Returns the categorical crossentropy between a true distribution and a
  predicted distribution.
  """
  # Note the corrective factor here to avoid division by zero in log:
  return -np.dot(true, np.log(pred + 1e-12))

def avg_cat_crossent(batch_true, batch_predicted):
  """
  Returns the average categorical crossentropy between paired members of the
  given batches.
  """
  result = 0
  for i in range(batch_true.shape[0]):
    result += cat_crossent(batch_true[i], batch_predicted[i])
  return result/batch_true.shape[0]

def avg_rmse(batch_true, batch_predicted):
  """
  Returns the average RMSE between paired members of the given batches.
  """
  result = 0
  for i in range(batch_true.shape[0]):
    result += np.sqrt(np.mean(np.power(batch_true[i] - batch_predicted[i], 2)))
  return result / batch_true.shape[0]

def rate_lstm_novelty(params, model, context, fragment):
  """
  Takes a context of at least window_size (raw text) as well as a fragment
  (also text) and using the context to spin up the network, predicts each
  character of the fragment in sequence, averaging the categorical crossentropy
  error-per-character to compute a novelty value for the fragment.
  """
  vec = vectorize.vectorize(params, context, pad=False)
  result = 0
  ivs = []
  trues = []
  for i in range(len(fragment)):
    iv = vec[-params["window_size"]:]
    ivs.append(iv)
    nv = vectorize.vectorize(params, fragment[i], pad=False)
    trues.append(nv[0])
    vec = np.append(vec, nv, axis=0)

  preds = np.array(model.predict(np.array(ivs), verbose=0), dtype=np.float64)
  trues = np.array(trues, dtype=np.float64)

  # TODO: Which of these?
  return avg_cat_crossent(trues, preds)
  #return avg_rmse(trues, preds)

def rate_cnn_novelty(params, model, fragment):
  """
  Takes a fragment of window_size raw text and tries to reconstruct it,
  averaging the categorical crossentropy error-per-character to compute a
  novelty value for the fragment.
  """
  vec = vectorize.vectorize(params, fragment, pad=False)
  result = 0
  trues = vec

  preds = model.predict(np.reshape(vec, (1,) + vec.shape), verbose=0)[0]
  trues = np.array(trues, dtype=np.float64)

  # TODO: Which of these?
  return avg_cat_crossent(trues, preds)
  #return avg_rmse(trues, preds)


@dep.task(("params", "lstm-model-final", "sentences"), "lstm-rated")
def lstm_rate_all(params, model, sentences):
  rated = []
  for i, (ctx, st) in enumerate(sentences):
    if len(st) == 0:
      continue
    utils.prbar(i/len(sentences), interval=6)
    nv = rate_lstm_novelty(params, model, ctx, st)
    rated.append((nv, "{}|{}".format(ctx, st)))

  utils.prdone()
  print("  ...done rating sentences.")

  rated = sorted(rated)
  return rated

@dep.task(
  ("params", "lstm-model-streaming-final", "input-stream"),
  "lstm-rated-stream",
  ("ephemeral",)
)
def lstm_stream_ratings(params, model, stream):
  ws = params["window_size"]
  def generate():
    for line in stream:
      nv = rate_lstm_novelty(params, model, STX * ws, line)
      yield (nv, line)

  return generate()

@dep.task(("params", "cnn-model-final", "sentences"), "cnn-rated")
def cnn_rate_all(params, model, sentences):
  """
  Rates a batch of sentences and returns a list of rating, 
  """
  print("Rating {} sentences...".format(len(sentences)))

  ws = params["window_size"]
  rated = []

  for i, (ctx, st) in enumerate(sentences):
    if len(st) == 0:
      continue
    utils.prbar(i/len(sentences), interval=6)
    chunks = []
    if len(st) < ws:
      chunks = [ ctx[len(st) - ws:] + st ]
      used = ctx[len(st) - ws:] + st
    else:
      chunks = [
        st[i:i+ws] for i in range(len(st) - ws + 1)
      ]
      used = st
    nv = np.mean([rate_cnn_novelty(params, model, c) for c in chunks])
    rated.append((nv, used))

  utils.prdone()
  print("  ...done rating sentences.")

  rated = sorted(rated)
  return rated

@dep.template_task(("{word}-rated",), "{word}-extremes")
def show_extremes(match, rated):
  """
  Shows the top-5 and bottom-5 rated sentences from a rated list.
  """
  result = ""

  boring = rated[:5]
  interesting = rated[-5:]

  result += "Least-novel:\n"
  for r, st in boring:
    st = utils.reflow(st)
    result += "{:.3g}: {}\n".format(r, st)

  result += "---\n"
  result += "Most-novel:\n"
  for r, st in interesting:
    st = utils.reflow(st)
    result += "{:.3g}: '{}'\n".format(r, st)

  return result

@dep.task(
  ("params", "lstm-rated-stream", "stream-epoch-count-estimate"),
  "lstm-streaming-extremes"
)
def stream_extremes(params, rated_stream, count_estimate):
  n = params["n_extremes"]
  top = []
  bot = []
  print("Finding extreme-rated examples...")
  for prog in range(count_estimate):
    utils.prbar(prog / count_estimate)
    rating, line = next(rated_stream)
    for i in range(n):
      if len(top) <= i:
        top.append((rating, line))
        break
      elif rating == top[i][0] and line == top[i][1]:
        break # duplicate
      elif rating > top[i][0]:
        top.insert(i, (rating, line))
        break
    if len(top) > n:
      top = top[:n]

    for i in range(n):
      if len(bot) <= i:
        bot.append((rating, line))
        break
      elif rating == bot[i][0] and line == bot[i][1]:
        break # duplicate
      elif rating < bot[i][0]:
        bot.insert(i, (rating, line))
        break
    bot = bot[:n]

  utils.prdone()
  print("  ...done finding extremes.")

  result = ""

  result += "Least-novel:\n"
  for r, st in bot:
    st = utils.reflow(st)
    result += "{:.3g}: {}\n".format(r, st)

  result += "---\n"
  result += "Most-novel:\n"
  for r, st in top:
    st = utils.reflow(st)
    result += "{:.3g}: '{}'\n".format(r, st)

  return result
