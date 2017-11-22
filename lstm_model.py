#!/usr/bin/env python3
"""
lstm_model.py

Trains an LSTM model and uses it to rate novelty of sentences.

Based on:

https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

Copyright (C) 2017 Peter Mawhorter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import random
import pickle

import numpy as np
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
#from keras.layers import Dropout
#from keras.layers import LSTM
#from keras.callbacks import ModelCheckpoint
#from keras.utils import np_utils

import utils

MAX_UNICHR = 0x10ffff

DEFAULT_PARAMS = {
  "input_directory": "data",
  "window_size": 64,
  "training_window_step": 1,
  "epochs": 10,
  "batch_size": 128,
  "models_dir": "models",
  "objects_dir": "objects",
}

STX = '\u0002'
ETX = '\u0003'
SUB = '\u001a'

def load_data(params):
  """
  Loads the data from the designated input directory, returning a list of
  strings (one for each document).
  """
  print("Loading data...")
  text_files = []
  for dp, dn, files in os.walk(params["input_directory"]):
    for f in files:
      if f.endswith(".txt"):
        text_files.append(os.path.join(dp, f))

  print("Loading from files:")
  texts = []
  for f in text_files:
    print("  " + f)
    with open(f, 'r') as fin:
      texts.append(utils.reflow(fin.read()))

  print("  ...done loading data.")
  return texts


def charmap(params, text):
  """
  Takes some text and creates a character -> integer mapping for it, returning
  a dictionary.

  The STX (\\u0002 -> '\u0002'), ETX (\\u0003 -> '\u0003'), and SUB (\\u001a ->
  '\u001a') characters are added to the map to be used as front padding, end
  padding, and unknown-character-replacement.
  """
  chars = sorted(list(set(text + STX + ETX + SUB)))
  return { c : i for i, c in enumerate(chars) }

def vectorize_onehot(params, text, cmap, pad=False):
  """
  Turns some text into a one-hot encoded version, returning the encoded text as
  a two-dimensional numpy array where each row encodes a character from the
  text (and therefore sums to 1) and each column corresponds to a character.
  Uses the given character mapping (see the charmap function).

  If pad is given as True, encoded STX and ETX will be used to pad the string
  by window_size on either side.
  """
  padding = params["window_size"] if pad else 0
  fpad = cmap.get(STX, cmap.get(SUB, 0))
  epad = cmap.get(ETX, cmap.get(SUB, 0))

  def onehot(i):
    return [0] * i + [1] + [0] * (len(cmap) - i - 1)
  
  return np.array(
    [onehot(fpad)] * padding + [
      onehot(cmap.get(c, cmap.get(SUB, 0)))
        for c in text
    ] + [onehot(epad)] * padding,
    dtype=np.bool
  )

def de_vectorize_onehot(params, vec, cmap):
  """
  Converts a 1-hot encoding as returned by vectorize back into characters using
  the given character map. Uses argmax to find most-active member of each row
  if the input isn't a true 1-hot encoding. If the given character map is
  deficient, the SUB character will be used for missing values.
  """
  result = ''
  rmap = {v: k for (k, v) in cmap.items()}
  for row in vec:
    idx = np.argmax(row)
    result += rmap.get(idx, SUB)

  return result

#def univec(c):
#  """
#  Translates a character into a binary vector using UTF-32 bits.
#  """
#  result = []
#  o = ord(c)
#  for sh in range(32):
#    result.append(((1 << sh) & o)>>sh)
#  return result
#
#def unichr(v):
#  """
#  Translates a binary vector into a character.
#  """
#  rord = 0
#  for i, b in enumerate(v):
#    rord |= (b > 0.5) << i
#
#  if rord > MAX_UNICHR:
#    return SUB
#
#  return chr(rord)

def onehot(i, n):
  return [0] * i + [1] + [0] * (n - i - 1)

def univec(c):
  """
  Translates a character into a binary vector starting from the Unicode
  codepoint and concatenating the one-hot encodings of each base-32 digit.
  """
  result = []
  o = ord(c)
  for i in range(5): # 32**5 > 0x10ffff
    result.extend(onehot(o % 32, 32))
    o //= 32
  return result

def unichr(v):
  """
  Translates a binary vector into a character. Uses argmax on each 32-entry
  chunk, so works even if the vector isn't purely binary.
  """
  rord = 0
  denom = 1
  for i in range(5):
    rord += denom * np.argmax(v[:32])
    v = v[32:]
    denom *= 32

  if rord > MAX_UNICHR:
    return SUB
  else:
    return chr(rord)


def vectorize(params, text, pad=False):
  """
  Turns some text into a bit-string encoded version, returning the encoded text
  as a two-dimensional numpy array where each row encodes a character from the
  text and each column corresponds to a character. UTF-32 codepoint bits are
  used as the encoding.

  If pad is given as True, encoded STX and ETX will be used to pad the string
  by window_size on either side.
  """
  padding = params["window_size"] if pad else 0
  
  return np.array(
    [univec(STX)] * padding + [
      univec(c) for c in text
    ] + [univec(ETX)] * padding,
    dtype=np.bool
  )

def de_vectorize(params, vec):
  """
  Converts a bit-array encoding as returned by vectorize back into characters
  using the given character map. Uses a threshold of 0.5 in case the input
  isn't really binary.
  """
  result = ''
  for row in vec:
    result += unichr(row)

  return result

def build_model(params):
  """
  Builds the Keras model.
  """
  encsize = len(univec(' '))
  histsize = params["window_size"] * encsize
  model = Sequential()
  model.add(LSTM(128, input_shape=(params["window_size"], encsize)))
  model.add(Activation("relu"))
  model.add(Dense(encsize))
  model.add(Activation("relu"))
  model.add(Dense(encsize))
  #model.add(Activation("relu"))
  model.add(Activation("softmax"))

  # TODO: Model specifics

  optimizer = RMSprop(lr=0.01)
  model.compile(loss="categorical_crossentropy", optimizer=optimizer)
  #model.compile(loss="mean_squared_error", optimizer=optimizer)

  return model

def sample(preds, temperature=1.0):
  """
  Helper function to sample a probability distribution. Taken from:
  https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
  """
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)


def generate(params, model, seed='', n=80):
  """
  Generates some text using the given model starting from the given seed. N
  specifies the number of additional characters to generate. If no seed is
  given, an STX block is used. The seed must be at least as long as the window
  size, or it will be pre-padded with STX characters.

  The temperature value controls how predictions are selected from the
  probability distribution returned by the network (see the sample function).
  """
  if len(seed) < params["window_size"]:
    seed = STX * (params["window_size"] - len(seed)) + seed
  result = ''
  vec = vectorize(params, seed, pad=False)
  for i in range(n):
    iv = vec[-params["window_size"]:]
    pred = model.predict(np.reshape(iv, (1,) + iv.shape), verbose=0)[0]
    nc = unichr(pred)
    result += nc
    nv = vectorize(params, nc, pad=False)
    vec = np.append(vec, nv, axis=0)

  return result

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
    result += sqrt(np.mean(np.power(batch_true[i] - batch_predicted[i], 2)))
  return result / batch_true.shape[0]

def rate_novelty(params, model, cmap, context, fragment):
  """
  Takes a context of at least window_size (raw text) as well as a fragment
  (also text) and using the context to spin up the network, predicts each
  character of the fragment in sequence, averaging the categorical crossentropy
  error-per-character to compute a novelty value for the fragment.
  """
  vec = vectorize(params, context, pad=False)
  rmap = {v: k for (k, v) in cmap.items()}
  result = 0
  ivs = []
  trues = []
  for i in range(len(fragment)):
    iv = vec[-params["window_size"]:]
    ivs.append(iv)
    nv = vectorize(params, fragment[i], pad=False)
    trues.append(nv[0])
    vec = np.append(vec, nv, axis=0)

  preds = np.array(model.predict(np.array(ivs), verbose=0), dtype=np.float64)
  trues = np.array(trues, dtype=np.float64)

  # TODO: Which of these?
  return avg_cat_crossent(trues, preds)
  #return avg_rmse(trues, preds)

def save_model(params, model, model_name):
  """
  Saves the given model to disk for retrieval using load_model.
  """
  model.save(os.path.join(params["models_dir"], model_name) + ".h5")

def load_model(params, model_name):
  """
  Loads the given model from disk. Returns None if no model with that name
  exists.
  """
  fn = os.path.join(params["models_dir"], model_name) + ".h5"
  if os.path.exists(fn):
    return keras.models.load_model(fn)
  else:
    return None


def save_object(params, obj, name):
  """
  Uses pickle to save the given object to a file.
  """
  fn = os.path.join(params["objects_dir"], name) + ".pkl"
  with open(fn, 'wb') as fout:
    pickle.dump(obj, fout)

def load_object(params, name):
  """
  Uses pickle to load the given object from a file. If the file doesn't exist,
  returns None.
  """
  fn = os.path.join(params["objects_dir"], name) + ".pkl"
  if os.path.exists(fn):
    with open(fn, 'rb') as fin:
      return pickle.load(fin)
  else:
    return None


@utils.default_params(DEFAULT_PARAMS)
def main(**params):
  """
  Main program. Loads data, vectorizes it, and then trains a network to
  reproduce it.
  """
  try:
    os.mkdir(params["models_dir"])
  except FileExistsError:
    pass

  try:
    os.mkdir(params["objects_dir"])
  except FileExistsError:
    pass

  texts = load_data(params)

  print("Creating charmap...")
  cm = charmap(params, ''.join(texts))

  cached = load_model(params, "lstm-final")
  if cached:
    print("Loading trained model...")
    model = cached
    print("...done loading model.")
  else:
    print("Compiling model...")
    model = load_model(params, "lstm-fresh") or build_model(params)
    save_model(params, model, "lstm-fresh")

    print("Vectorizing data...")
    vectors = []
    for t in texts:
      vectors.append(vectorize(params, t, pad=True))

    print(
      "Total characters: {}\nVocabulary: {}\n{}".format(
        sum(len(t) for t in texts),
        len(cm),
        ''.join(
          sorted([k for k in cm])
        ).replace('\n', r'\n').replace('\r', r'\r')
      )
    )

    print("Distilling training sequences...")
    train = []
    target = []
    for tv in vectors:
      for i in range(
        params["window_size"] // 2,
        len(tv) - params["window_size"],
        params["training_window_step"]
      ):
        train.append(tv[i:i + params["window_size"]])
        target.append(tv[i+params["window_size"]])

    train = np.array(train, dtype=np.bool)
    target = np.array(target, dtype=np.bool)

    print("Training model...")
    for epoch in range(1, params["epochs"]+1):
      print()
      print('-'*80)
      print("Epoch {}".format(epoch))
      cached = load_model(params, "lstm-epoch-{}".format(epoch))
      if cached:
        model = cached
      else:
        model.fit(train, target, batch_size=params["batch_size"], epochs=1)
        save_model(params, model, "lstm-epoch-{}".format(epoch))

      start_from = random.choice(vectors)
      start_index = random.randint(
        params["window_size"]//2,
        len(start_from) - params["window_size"]*2
      )

      # TODO: Not this?
      seed = start_from[start_index:start_index + params["window_size"]]
      generated = de_vectorize(params, seed)
      print("--- generating...".format(generated))
      pre = generated;
      gen = generate(
        params,
        model,
        seed=generated,
        n=80*4
      )
      gen = utils.reflow(pre + "|" + gen)
      print(gen)
      print("---")

    print("...done with training.")
    save_model(params, model, "lstm-final")

  print("Separating sentences...")
  sentences = []
  for t in texts:
    si = 0
    ei = 0
    done = False
    while not done:
      try:
        ei = t.index('.', si+1)
      except ValueError:
        ei = len(t)
        done = True
      if si == 0:
        sentences.append((STX*params["window_size"], t[si:ei]))
      elif si < params["window_size"]:
        pre = t[:si]
        pre = STX * (params["window_size"] - len(pre)) + pre
        sentences.append((pre, t[si:ei]))
      else:
        sentences.append((t[si-params["window_size"]:si], t[si:ei]))
      si = ei

  print("Rating {} sentences...".format(len(sentences)))
  cached = load_object(params, "lstm-rated")
  if cached:
    print("  ...loaded saved ratings.")
    rated = cached
  else:
    rated = []
    for i, (ctx, st) in enumerate(sentences):
      if len(st) == 0:
        continue
      utils.prbar(i/len(sentences), interval=5)
      nv = np.mean(rate_novelty(params, model, cm, ctx, st))
      rated.append((nv, ctx, st))
    utils.prdone()
    save_object(params, rated, "lstm-rated")
    print("  ...done rating sentences.")

  rated = sorted(rated, key=lambda abc: (abc[0], abc[1], abc[2]))
  boring = rated[:5]
  interesting = rated[-5:]

  print("---")
  print("Least-novel:")
  for r, ctx, st in boring:
    st = utils.reflow(st)
    if st.startswith(". "):
      st = st[2:] + "."
    print("{:.3g}: {}".format(r, st))

  print("---")
  print("Most-novel:")
  for r, ctx, st in interesting:
    st = utils.reflow(st)
    if st.startswith(". "):
      st = st[2:] + "."
    print("{:.3g}: '{}'".format(r, st))

if __name__ == "__main__":
  main()
