#!/usr/bin/env python3
"""
main.py

Trains a neural network model and uses it to rate novelty of sentences.

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

import dep
import sys
import copy

# target-defining modules:
import load
import lstm_model
import cnn_model
import generate
import rate

DEFAULT_PARAMS = {
  "input_directory": "data",
  "default_streaming_epoch_count": 100000,
  "exact_streaming_epoch_count": None,
  "stream_line_length_guess": 320,
  "streaming_source": "data/chat/test.txt",
  "stream_chunk_size": 1024*1024*256, # up to 256 MB in memory at once
  "stream_sleep": 1/1000000, # one microsecond
  "generate_size": 80,
  "window_size": 32,
  "training_window_step": 1,
  "conv_sizes": ((48, 4), (32, 3)),
  "dense_sizes": (512, 256, 128),
  "regularization": 1e-5,
  "encoded_layer_name": "final_encoder",
  "decoded_layer_name": "final_decoder",
  "epochs": 10,
  "batch_size": 128,
  "superbatch_size": 128,
  "models_dir": "models",
  "objects_dir": "objects",
  "gen_length": 80*4,
  "n_extremes": 100,
}

#dep.add_alias("default", "lstm-rated")

dep.add_alias("default", "lstm-streaming-extremes")

def main(*targets):
  """
  Main program. Builds the desired targets.
  """
  if not targets:
    targets = ["default"]

  for t in targets:
    d1 = '-' * (max(0, 80 - 2 - len(t))//2)
    d2 = '-' * (80 - 2 - len(t) - len(d1))
    print("{} {} {}".format(d1, t, d2))
    print(dep.create(t)[1])


if __name__ == "__main__":
  args = sys.argv[1:]
  targets = []
  params = copy.deepcopy(DEFAULT_PARAMS)
  for a in args:
    if a.startswith("--"):
      spl = a.index('=')
      key = a[2:spl]
      val = a[spl+1:]
      if key in params:
        if type(params[key]) == str:
          params[key] = val
        else:
          params[key] = eval(val)
        continue # not a target
    targets.append(a)

  dep.add_object(params, "params")

  main(*targets)
