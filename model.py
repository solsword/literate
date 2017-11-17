#!/usr/bin/env python3
"""
model.py

Builds model.

Based on:

https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
"""

import os
import random

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
#from keras.layers import Dropout
#from keras.layers import LSTM
#from keras.callbacks import ModelCheckpoint
#from keras.utils import np_utils

import utils

DEFAULT_PARAMS = {
  "input_directory": "data",
  "window_size": 64,
  "training_window_step": 1,
  "epochs": 10,
  "batch_size": 128,
  "models_dir": "models",
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

def vectorize(params, text, cmap, pad=False):
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

def de_vectorize(params, vec, cmap):
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

def build_model(params, cmap):
  """
  Builds the Keras model. Needs the character map to know what size inputs to
  take.
  """
  model = Sequential()
  model.add(LSTM(128, input_shape=(params["window_size"], len(cmap))))
  model.add(Dense(len(cmap)))
  model.add(Activation('softmax'))

  optimizer = RMSprop(lr=0.01)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer)

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


def generate(params, model, cmap, seed='', n=80, temperature=1.0):
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
  rmap = {v: k for (k, v) in cmap.items()}
  result = ''
  vec = vectorize(params, seed, cmap, pad=False)
  for i in range(n):
    iv = vec[-params["window_size"]:]
    preds = model.predict(np.reshape(iv, (1,) + iv.shape), verbose=0)[0]
    nc = rmap.get(sample(preds, temperature), SUB)
    result += nc
    nv = vectorize(params, nc, cmap, pad=False)
    vec = np.append(vec, nv, axis=0)

  return result

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


@utils.default_params(DEFAULT_PARAMS)
def main(**params):
  """
  Main program. Loads data, vectorizes it, and then trains a network to
  reproduce it.
  """
  texts = load_data(params)
  print("Vectorizing data...")
  cm = charmap(params, ''.join(texts))
  vectors = []
  for t in texts:
    vectors.append(vectorize(params, t, cm, pad=True))

  print(
    "Total characters: {}\nVocabulary: {}\n{}".format(
      sum(len(t) for t in texts),
      len(cm),
      ''.join(sorted([k for k in cm])).replace('\n', r'\n').replace('\r', r'\r')
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

  print("Compiling model...")
  model = load_model(params, "fresh") or build_model(params, cm)
  save_model(params, model, "fresh")

  print("Training model...")
  for epoch in range(1, params["epochs"]+1):
    print()
    print('-'*80)
    print("Epoch {}".format(epoch))
    cached = load_model(params, "epoch-{}".format(epoch))
    if cached:
      model = cached
    else:
      model.fit(train, target, batch_size=params["batch_size"], epochs=1)
      save_model(params, model, "epoch-{}".format(epoch))

    start_from = random.choice(vectors)
    start_index = random.randint(
      params["window_size"]//2,
      len(start_from) - params["window_size"]*2
    )

    for diversity in [0.2, 0.5, 1.0, 1.2]:
      print()
      print("--- diversity: {}".format(diversity))

      seed = start_from[start_index:start_index + params["window_size"]]
      generated = de_vectorize(params, seed, cm)
      print("--- generating from: '{}'".format(generated))
      print(generated, end="")
      gen = generate(
        params,
        model,
        cm,
        seed=generated,
        n=80*4,
        temperature=diversity
      )
      print(gen)
      print("---")

  save_model(params, model, "final")

if __name__ == "__main__":
  main()
