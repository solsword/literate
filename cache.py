"""
cache.py

Code for saving/loading models and objects.
"""

import os
import time
import pickle

import keras

def save_model(cache_dir, model, model_name):
  """
  Saves the given model to disk for retrieval using load_model.
  """
  model.save(os.path.join(cache_dir, model_name) + ".h5")

def load_model(cache_dir, model_name):
  """
  Loads the given model from disk. Raises a ValueError if the target doesn't
  exist. Returns a (timestamp, value) pair.
  """
  fn = os.path.join(cache_dir, model_name) + ".h5"
  if os.path.exists(fn):
    ts = os.path.getmtime(fn)
    return (ts, keras.models.load_model(fn))
  else:
    raise ValueError(
      "Model '{}' isn't stored in directory '{}'.".format(
        model_name,
        cache_dir
      )
    )


def save_object(cache_dir, obj, name):
  """
  Uses pickle to save the given object to a file.
  """
  fn = os.path.join(cache_dir, name) + ".pkl"
  with open(fn, 'wb') as fout:
    pickle.dump(obj, fout)

def load_object(cache_dir, name):
  """
  Uses pickle to load the given object from a file. If the file doesn't exist,
  raises a ValueError. Returns a (timestamp, value) pair.
  """
  fn = os.path.join(cache_dir, name) + ".pkl"
  if os.path.exists(fn):
    ts = os.path.getmtime(fn)
    with open(fn, 'rb') as fin:
      return (ts, pickle.load(fin))
  else:
    raise ValueError(
      "Object '{}' isn't stored in directory '{}'.".format(
        name,
        cache_dir
      )
    )

def save_any(cache_dir, obj, name):
  """
  Selects save_object or save_model automatically.
  """
  if isinstance(obj, (keras.models.Sequential, keras.models.Model)):
    save_model(cache_dir, obj, name)
  else:
    save_object(cache_dir, obj, name)

def load_any(cache_dir, name):
  """
  Attempts load_object and falls back to load_model.
  """
  try:
    return load_object(cache_dir, name)
  except:
    return load_model(cache_dir, name)

def check_time(cache_dir, name):
  """
  Returns just the modification time for the given object (tries pickle first
  and then h5). Returns None if the file does not exist.
  """
  fn = os.path.join(cache_dir, name) + ".pkl"
  if os.path.exists(fn):
    return os.path.getmtime(fn)
  else:
    fn = os.path.join(cache_dir, name) + ".h5"
    if os.path.exists(fn):
      return os.path.getmtime(fn)
    else:
      return None
