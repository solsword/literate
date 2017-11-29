"""
cnn_model.py

Code for building a CNN model.
"""

import os
import random

import numpy as np
import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from keras.regularizers import l1

import utils
import dep

import vectorize

@dep.task(("cnn-model-fresh",), "cnn-model-summary")
def summarize_model(model):
  return model.summary()

@dep.task(("params",), "cnn-model-fresh")
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


@dep.task(("params", "texts"), "cnn-examples")
def separate_examples(params, texts):
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

  return examples

@dep.iter_task(
  ("params", "cnn-examples", "cnn-model-epoch-{iter}"),
  "cnn-model-epoch-{next}",
  ("volatile",)
)
def train_one_epoch(epoch, params, examples, model):
  """
  Trains the CNN model for a single epoch.
  """
  print()
  print('-'*80)
  print("Epoch {}".format(epoch))
  sbs = params["superbatch_size"] * params["batch_size"]
  for bs in range(0, len(examples), sbs):
    utils.prbar(bs / len(examples), interval=1)
    batch = examples[bs:bs + sbs]
    bvec = np.array(
      [
        vectorize.vectorize(params, ex, pad=False)
          for ex in batch
      ],
      dtype=np.bool
    )
    model.fit(bvec, bvec, batch_size=params["batch_size"], epochs=1, verbose=0)
    del batch
    del bvec
  utils.prdone()
  print("...done with epoch {}.".format(epoch))
  return model

dep.add_alias("cnn-model-epoch-start", "cnn-model-fresh")
dep.add_alias("cnn-model-final", "cnn-model-epoch-10")
