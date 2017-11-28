"""
load.py

Data loading and basic processing.
"""

import os

import utils
import dep

STX = '\u0002'
ETX = '\u0003'

@dep.task(("params",), "texts")
def load_data(params):
  """
  Loads the data from the designated input directory, returning a list of
  strings (one for each document).
  """
  print("Loading data...")
  text_files = []
  print(params)
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


@dep.task(("params","texts"), "sentences")
def separate_sentences(params, texts):
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

  return sentences
