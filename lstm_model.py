"""
lstm_model.py

Code for building an LSTM model.
"""

import os
import random
import pickle

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
#from keras.layers import Dropout
#from keras.layers import LSTM
#from keras.callbacks import ModelCheckpoint
#from keras.utils import np_utils

from . import utils
from . import dep

from . import vectorize

STX = '\u0002'
ETX = '\u0003'

@dep.task(("lstm-model-fresh",), "lstm-model-summary")
def summarize_model(model):
  return model.summary()

@dep.task(("params",), "lstm-model-fresh")
def build_model(params):
  """
  Builds the Keras LSTM model.
  """
  encsize = len(vectorize.univec(' '))
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

@dep.task(("params", "texts"), "lstm-examples")
def separate_examples(params, texts):
  ws = params["window_size"]
  examples = []
  for i, t in enumerate(texts):
    print("[{}/{}]".format(i, len(texts)))
    for ed in range(ws//2, len(t) + ws//2):
      utils.prbar((ed - ws//2) / len(t), interval=19)
      st = ed - (ws + 1)
      pre = ''
      post = ''
      if st < 0:
        pre = STX * (-st)
        st = 0
      if ed > len(t):
        post = ETX * (ed - len(t))
        ed = len(t)

      ex = pre + t[st:ed] + post
      examples.append((ex[:-1], ex[-1]))

    utils.prdone()

  return examples

@dep.task(("params", "input-stream"), "lstm-example-stream", ("ephemeral",))
def stream_examples(params, lines):
  """
  Takes a line-by-line input stream and generates an examples stream from it.
  """
  ws = params["window_size"]
  def generator():
    for line in lines: # infinite
      for ed in range(ws//2, len(line) + ws//2): # .5 window size on either end
        st = ed - (ws + 1)
        pre = ''
        post = ''
        if st < 0:
          pre = STX * (-st)
          st = 0
        if ed > len(line):
          post = ETX * (ed - len(line))
          ed = len(line)
        
        ex = pre + line[st:ed] + post
        yield ( ex[:-1], ex[-1] )

  return generator()

@dep.iter_task(
  ("params", "lstm-examples", "lstm-model-epoch-{iter}"),
  "lstm-model-epoch-{next}",
  ("volatile",)
)
def train_one_epoch(epoch, params, examples, model):
  """
  Trains the LSTM model for a single epoch.
  """
  print()
  print('-'*80)
  print("Epoch {}".format(epoch))
  sbs = params["superbatch_size"] * params["batch_size"]
  for bs in range(0, len(examples), sbs):
    utils.prbar(bs / len(examples), interval=1)
    batch = examples[bs:bs + sbs]
    bivec = np.array( # batch input vector
      [
        vectorize.vectorize(params, ex[0], pad=False)
          for ex in batch
      ],
      dtype=np.bool
    )
    bevec = np.array( # batch expected vector
      [
        vectorize.vectorize(params, ex[1], pad=False)[0]
          for ex in batch
      ],
      dtype=np.bool
    )
    model.fit(
      bivec,
      bevec,
      batch_size=params["batch_size"],
      epochs=1,
      verbose=0
    )
    del batch
    del bivec
    del bevec
  utils.prdone()
  print("...done with epoch {}.".format(epoch))
  return model

dep.add_alias("lstm-model-epoch-start", "lstm-model-fresh")
dep.add_alias("lstm-model-final", "lstm-model-epoch-10")

@dep.iter_task(
  (
    "params",
    "lstm-example-stream",
    "stream-epoch-count-estimate",
    "lstm-model-streaming-epoch-{iter}"
  ),
  "lstm-model-streaming-epoch-{next}",
  ("volatile",)
)
def train_one_epoch_streaming(
  epoch,
  params,
  example_stream,
  count_estimate,
  model
):
  """
  Trains the LSTM model for a single epoch from an example stream.
  """
  print()
  print('-'*80)
  print("Epoch {}".format(epoch))
  sbs = params["superbatch_size"] * params["batch_size"]
  processed = 0
  for bs in range(0, count_estimate, sbs):
    utils.prbar(bs / count_estimate, interval=1)
    batch = []
    for i in range(bs, min(count_estimate, bs + sbs)):
      processed += 1
      batch.append(next(example_stream))
    bivec = np.array( # batch input vector
      [
        vectorize.vectorize(params, ex[0], pad=False)
          for ex in batch
      ],
      dtype=np.bool
    )
    bevec = np.array( # batch expected vector
      [
        vectorize.vectorize(params, ex[1], pad=False)[0]
          for ex in batch
      ],
      dtype=np.bool
    )
    model.fit(
      bivec,
      bevec,
      batch_size=params["batch_size"],
      epochs=1,
      verbose=0
    )
    del batch
    del bivec
    del bevec
  utils.prdone()
  print(
    "...done with epoch {} ({} examples processed).".format(
      epoch,
      processed
    )
  )
  return model

dep.add_alias("lstm-model-streaming-epoch-start", "lstm-model-fresh")
dep.add_alias("lstm-model-streaming-final", "lstm-model-streaming-epoch-10")
