#!/usr/bin/env python3
"""
cnn_model.py

Trains a CNN model and uses it to rate novelty of sentences.

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
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from keras.regularizers import l1

import utils

MAX_UNICHR = 0x10ffff
PRACTICAL_MAX_UNICHR = 0x2ffff

DEFAULT_PARAMS = {
  "input_directory": "data",
  "window_size": 32,
  "training_window_step": 1,
  "conv_sizes": [(48, 4), (32, 3)],
  "dense_sizes": [512, 256, 128],
  "regularization": 1e-5,
  "encoded_layer_name": "final_encoder",
  "decoded_layer_name": "final_decoder",
  "epochs": 10,
  "batch_size": 128,
  "superbatch_size": 128,
  "models_dir": "models",
  "objects_dir": "objects",
  "gen_length": 80*4,
}

NUL = '\u0000'
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

def univec(c):
  """
  Translates a character into a binary vector starting from the Unicode
  codepoint and concatenating the one-hot encodings of each base-32 digit. Only
  allows up to 4 base-32 digits, which is enough to cover every codepoint in
  the basic multilingual plan, the supplementary multilingual plane, and the
  supplementary ideographic plane. Characters outside this range are replaced
  by the substitute character.
  """
  result = np.array([0]*(32*4), dtype=bool)
  o = ord(c)
  if o > PRACTICAL_MAX_UNICHR:
    o = ord(SUB)
  # unrolled loop x4; 32**5 > 0x10ffff
  result[32*0 + (o % 32)] = 1
  o //= 32
  result[32*1 + (o % 32)] = 1
  o //= 32
  result[32*2 + (o % 32)] = 1
  o //= 32
  result[32*3 + (o % 32)] = 1
  o //= 32
  return result

def unichr(v):
  """
  Translates a binary vector into a character. Uses argmax on each 32-entry
  chunk, so works even if the vector isn't purely binary.
  """
  rord = 0
  denom = 1
  for i in range(4):
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
  input_str = Input(shape=(params["window_size"], encsize));
  network = input_str

  # Encoding convolution:
  for i, (units, width) in enumerate(params["conv_sizes"]):
    network = Conv1D(
      units,
      width,
      activation='relu',
      padding='valid',
      name = "conv_encoder_{}-{}x{}".format(i, units, width)
    )(network)
    network = MaxPooling1D(
      pool_size=2,
      padding='valid',
      name="pool_{}".format(i)
    )(network)

  conv_shape = network._keras_shape[1:]
  network = Flatten(name="flatten")(network)
  flat_length = network._keras_shape[-1]

  # Encoding dense layers:
  for i, sz in enumerate(params["dense_sizes"]):
    if i == len(params["dense_sizes"])-1:
      reg = l1(params["regularization"])
      name = params["encoded_layer_name"]
    else:
      reg = None
      name = "dense_encoder_{}".format(i)

    network = Dense(
      sz,
      activation='relu',
      activity_regularizer=reg,
      name=name
    )(network)

  # Decoding dense layers:
  for i, sz in enumerate(list(reversed(params["dense_sizes"]))[1:]):
    network = Dense(
      sz,
      activation='relu',
      name="dense_decoder_{}".format(i)
    )(network)

  # size appropriately:
  network = Dense(flat_length, activation='relu', name="dense_grow")(network)
  network = Reshape(conv_shape, name="unflatten")(network)

  for i, (units, width) in enumerate(reversed(params["conv_sizes"])):
    network = UpSampling1D(size=2, name="upsample_{}".format(i))(network)
    network = Conv1D(
      units,
      width,
      activation='relu',
      padding='valid',
      name="conv_decoder_{}-{}x{}".format(i, units, width)
    )(network)

  network = Flatten(name="final_flatten")(network)
  network = Dense(
    params["window_size"] * encsize,
    activation='relu',
    name=params["decoded_layer_name"],
  )(network)
  network = Reshape(
    (params["window_size"], encsize),
    name="final_reshape"
  )(network)

  model = Model(input_str, network)
  model.compile(
    optimizer='adagrad',
    loss='mean_squared_error'
  )

  return model

def generate(params, model, seed='', n=80):
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

def itergen(params, model, limit=100):
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

def cat_crossent(true, pred):
  """
  Returns the categorical crossentropy between a true distribution and a
  predicted distribution.
  """
  # Note the corrective factor here to avoid division by zero in log:
  return -np.dot(true, np.log(pred + 1e-12))
  # TODO: Not this!
  #baseline = np.min(pred)
  #return -np.dot(true, np.log(pred - baseline + 1e-12))

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

def rate_novelty(params, model, fragment):
  """
  Takes a fragment of window_size raw text and tries to reconstruct it,
  averaging the categorical crossentropy error-per-character to compute a
  novelty value for the fragment.
  """
  vec = vectorize(params, fragment, pad=False)
  result = 0
  trues = vec

  preds = model.predict(np.reshape(vec, (1,) + vec.shape), verbose=0)[0]
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

  print("Total characters: {}".format(sum(len(t) for t in texts)))

  cached = load_model(params, "cnn-final")
  if cached:
    print("Loading trained model...")
    model = cached
    print("Model summary:")
    print(model.summary())
    print("...done loading model.")
  else:
    print("Compiling model...")
    model = load_model(params, "cnn-fresh") or build_model(params)
    save_model(params, model, "cnn-fresh")

    print("Model summary:")
    print(model.summary())

    print("Exploding training sequences...")
    cached = load_object(params, "cnn-examples")
    if cached:
      examples = cached
    else:
      ws = params["window_size"]
      examples = []
      for i, t in enumerate(texts):
        print("[{}/{}]".format(i, len(texts)))
        for ed in range(ws//2, len(t) + ws//2):
          utils.prbar((ed - ws//2) / len(t), interval=19)
          st = ed - ws
          pre = ''
          post = ''
          if st < 0:
            pre = STX * (-st)
            st = 0
          if ed > len(t):
            post = ETX * (ed - len(t))
            ed = len(t)

          ex = pre + t[st:ed] + post
          examples.append(ex)

        utils.prdone()

      save_object(params, examples, "cnn-examples")

    print("Training model...")
    for epoch in range(1, params["epochs"]+1):
      print()
      print('-'*80)
      print("Epoch {}".format(epoch))
      cached = load_model(params, "cnn-epoch-{}".format(epoch))
      if cached:
        model = cached
      else:
        sbs = params["superbatch_size"] * params["batch_size"]
        for bs in range(0, len(examples), sbs):
          utils.prbar(bs / len(examples), interval=1)
          batch = examples[bs:bs + sbs]
          bvec = np.array(
            [
              vectorize(params, ex, pad=False)
                for ex in batch
            ],
            dtype=np.bool
          )
          model.fit(bvec, bvec, batch_size=len(bvec), epochs=1, verbose=0)
          # TODO: Does this work?!?
          del batch
          del bvec
        utils.prdone()
        save_model(params, model, "cnn-epoch-{}".format(epoch))

      print("...done with training.")
      save_model(params, model, "cnn-final")


  cached = load_object(params, "cnn-generated")
  if cached:
    gen = cached
    print("Loading generated text...")
  else:
    print("Generating text...")
    gen = []
    for i in range(10):
      gen.append(utils.reflow(itergen(params, model)))
      utils.prbar(i/10)
    utils.prdone()
    save_object(params, gen, "cnn-generated")
  print("  ...results:")
  for g in gen:
    print(" ", g)

  print("Separating sentences...")
  sentences = []
  for t in texts:
    si = 0
    ei = 0
    while si < len(t):
      try:
        ei = t.index('.', si) + 1
      except ValueError:
        ei = len(t)
      if si == 0:
        sentences.append((STX*params["window_size"], t[si:ei]))
      elif si < params["window_size"]:
        if t[si] in ' \n':
          si += 1
        pre = t[:si]
        pre = STX * (params["window_size"] - len(pre)) + pre
        sentences.append((pre, t[si:ei]))
      else:
        if t[si] in ' \n':
          si += 1
        sentences.append((t[si-params["window_size"]:si], t[si:ei]))
      si = ei

  print("Rating {} sentences...".format(len(sentences)))
  cached = load_object(params, "cnn-rated")
  if cached:
    print("  ...loaded saved ratings.")
    rated = cached
  else:
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
      nv = np.mean([rate_novelty(params, model, c) for c in chunks])
      rated.append((nv, used))
    utils.prdone()
    save_object(params, rated, "cnn-rated")
    print("  ...done rating sentences.")

  rated = sorted(rated, key=lambda ab: (ab[0], ab[1]))
  boring = rated[:5]
  interesting = rated[-5:]

  print("---")
  print("Least-novel:")
  for r, chunks in boring:
    st = utils.reflow(''.join(chunks))
    print("{:.3g}: {}".format(r, st))

  print("---")
  print("Most-novel:")
  for r, chunks in interesting:
    st = utils.reflow(''.join(chunks))
    print("{:.3g}: '{}'".format(r, st))

if __name__ == "__main__":
  main()
