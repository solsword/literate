"""
load.py

Data loading and basic processing.
"""

import os
import time

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

@dep.task(("params",), "stream-epoch-count-estimate")
def estimate_stream_epoch_count(params):
  """
  Estimates the number of examples per epoch based on the size of the input
  file for stream-based training. Returns the default_streaming_epoch_count
  parameter value if the size of the input stream can't be computed (e.g., when
  it's an actual stream instead of a file). This estimate is generally an
  overestimate, because newlines are filtered out downstream but are counted
  towards file size. The stream_line_length_guess parameter is used to adjust
  the estimate, so this may become an underestimate if the average line length
  is greater than stream_line_length_guess. Across even a few epochs, the
  effect of over/under-training shouldn't be too bad, assuming the total number
  of examples is fairly high (which is the reason to use streaming in the first
  place).
  """
  src = params["streaming_source"]
  try:
    st = os.stat(src)
    sz = st.st_size
    return sz - int(sz / params["stream_line_length_guess"])
  except:
    return params["default_streaming_epoch_count"]

@dep.task(("params",), "input-stream", ("ephemeral",))
def stream_lines(params):
  """
  Sets up a streaming data source, returning a generator that yields one line
  at a time. The generator will yield indefinitely, repeating from the
  beginning once the source is exhausted. Newlines in the source are not
  returned as part of the lines generated; blank lines are skipped.
  """
  def generator():
    src = params["streaming_source"]
    with open(src, 'r') as fin:
      empty = False
      while True:
        bucket = fin.readlines(params["stream_chunk_size"])
        if sum(len(l) for l in bucket) < params["stream_chunk_size"]:
          # at end of file; may be false negative w/ multibyte characters but
          # won't be false positive.
          if fin.seekable():
            fin.seek(0) # back to the beginning!
          # otherwise just wait for more stuff to be put into the pipe
        empty = True
        for l in bucket:
          if len(l) > 1: # a character plus a newline
            empty = False
            yield l[:-1] # without the newline
        if empty:
          # if our bucket was empty, sleep here to avoid spinning
          time.sleep(params["stream_sleep"])
        # now loop to get the next bucket

  return generator()
