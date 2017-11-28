"""
dep.py

Python make-like object building/caching system.
"""

import os
import cache
import time
import regex as re # for same-name group overwriting
import collections

TARGET_ALIASES = {}
KNOWN_TARGETS = {}
TARGET_GENERATORS = collections.OrderedDict()

CACHED_VALUES = {}

CACHE_DIR = ".dep_cache"

DC1 = '\u0010'
DC2 = '\u0011'
DC3 = '\u0012'
DC4 = '\u0013'

def set_cache_dir(dr):
  """
  Sets the cache directory. Doesn't delete the old one (do that manually).
  """
  CACHE_DIR = dr
  try:
    os.mkdir(CACHE_DIR)
  except FileExistsError:
    pass

# Create the cache dir on import if needed:
set_cache_dir(CACHE_DIR)

def add_alias(alias, target):
  """
  Adds an alias that simply bridges between targets. Note that aliases take
  priority over actual targets, so be careful not to shadow stuff. Alias
  chaining does work, but can also create infinite loops.
  """
  global TARGET_ALIASES
  TARGET_ALIASES[alias] = target

def add_object(obj, target, flags=()):
  """
  Adds a task that simply returns the given object.
  """
  global KNOWN_TARGETS
  def offer():
    nonlocal obj
    return obj
  KNOWN_TARGETS[target] = ((), offer, flags)

def add_gather(inputs, output, flags=()):
  """
  Adds a task which simply gathers all of its dependencies into a list.
  """
  global KNOWN_TARGETS
  def gather(*inputs):
    return inputs
  KNOWN_TARGETS[output] = (inputs, gather, flags)

def task(inputs, output, flags=()):
  """
  A decorator for defining a task. Registers the decorated function as the
  mechanism for producing the declared target using the given inputs.
  """
  def decorate(function):
    global KNOWN_TARGETS
    KNOWN_TARGETS[output] = (inputs, function, flags)
    return function
  return decorate

def template_task(inputs, output, flags=()):
  """
  A decorator similar to task, but it generates targets by replacing named
  formatting groups within input/output strings with appropriate matches.
  """
  def decorate(function):
    global TARGET_GENERATORS

    slots = re.findall(r"(?<!{){[^{}]*}", output)

    plainslots = slots.count("{}")
    keyslots = set( sl[1:-1] for sl in slots if sl != "{}" )

    if plainslots + len(keyslots) > 16:
      raise ValueError("Too many slots (>16)!\n{}".format(output))

    # Encode indices using control character pairs
    digits = [DC1, DC2, DC3, DC4]
    plainrep = [ digits[i//4] + digits[i%4] for i in range(plainslots) ]
    keyrep = {}
    i = plainslots
    for k in keyslots:
      keyrep[k] = digits[i//4] + digits[i%4]
      i += 1

    keygroups = { k:  r"(?<" + k + r">.+)" for k in keyslots }

    tre = re.escape(output.format(*plainrep, **keyrep))
    for i in range(plainslots):
      tre = tre.replace(digits[i//4] + digits[i%4], r"(.+)")
    for k in keyrep:
      tre = tre.replace(keyrep[k], keygroups[k])

    def gen_target(name_match, stuff):
      inputs, function, flags = stuff

      gd = name_match.groupdict()
      inputs = [ inp.format(**gd) for inp in inputs ]
      def wrapped(*args, **kwargs):
        nonlocal function, name_match
        return function(name_match, *args, **kwargs)
      wrapped.__name__ = function.__name__
      return inputs, wrapped, flags

    TARGET_GENERATORS[tre] = (gen_target, (inputs, function, flags))
    return function
  return decorate

def iter_task(inputs, output, flags=()):
  """
  A decorator similar to task, but it generates targets by replacing {iter} and
  {next} within input/output strings with subsequent natural numbers.
  """
  def decorate(function):
    global TARGET_GENERATORS

    tre = re.escape(output.format(iter=DC1, next=DC2))
    tre = tre.replace(DC1, r"(?P<iter>[0-9]+)")
    tre = tre.replace(DC2, r"(?P<next>[0-9]+)")

    def gen_target(name_match, stuff):
      inputs, function, flags = stuff

      try:
        ival = int(name_match.group("iter"))
      except IndexError:
        ival = None
      try:
        nval = int(name_match.group("next"))
      except IndexError:
        nval = None

      if ival == None and nval != None:
        if nval <= 0:
          ival = "start"
        else:
          ival = nval - 1
      elif ival != None and nval == None:
        nval = ival + 1
      elif ival == None or nval == None:
        ival = "start"
        nval = 0

      inputs = [ inp.format(iter=ival, next=nval) for inp in inputs ]
      def wrapped(*args, **kwargs):
        nonlocal function, nval
        return function(nval, *args, **kwargs)
      wrapped.__name__ = function.__name__
      return inputs, wrapped, flags

    TARGET_GENERATORS[tre] = (gen_target, (inputs, function, flags))
    return function
  return decorate

class NotAvailable:
  pass

def get_cache_time(target):
  """
  Gets the cache time of the given target. Returns None if the target isn't
  cached anywhere.
  """
  if target in CACHED_VALUES:
    return CACHED_VALUES[target][0]
  else:
    return cache.check_time(CACHE_DIR, target)

def get_cached(target):
  """
  Fetches a cached object for the given target, or returns a special
  NotAvailable result. Returns a (timestamp, value) pair, with None as the time
  if the object isn't available.
  """
  if target in CACHED_VALUES:
    # in memory
    return CACHED_VALUES[target]
  else:
    try:
      # on disk
      return cache.load_any(CACHE_DIR, target)
    except:
      # must create
      return None, NotAvailable

def cache_value(target, value, flags):
  """
  Adds a value to the cache, also storing it to disk. Returns the timestamp of
  the newly-cached value.
  """
  cache.save_any(CACHE_DIR, value, target)
  ts = time.time()
  if "volatile" in flags: # Only save to disk
    try:
      del CACHED_VALUES[target]
    except:
      pass
  else:
    CACHED_VALUES[target] = (ts, value)
  return ts

def find_target(target):
  """
  Retrieves information (inputs, processing function, and flags) for the given
  target. Generates a target when necessary and possible.
  """
  while target in TARGET_ALIASES:
    target = TARGET_ALIASES[target]
  if target in KNOWN_TARGETS:
    # known target: return it
    return KNOWN_TARGETS[target]
  else:
    # try to find a generator that can handle it?
    for tre in TARGET_GENERATORS:
      m = re.match(tre, target)
      if m:
        gen, stuff = TARGET_GENERATORS[tre]
        try:
          return gen(m, stuff)
        except:
          pass

  # Not a known target and no generator matches:
  raise ValueError("Unknown target '{}'.".format(target))

def check_up_to_date(target):
  """
  Returns a timestamp for the given target after checking that all of its
  (recursive) perquisites are up-to-date. If missing and/or out-of-date values
  are found, new values are generated.
  """
  inputs, function, flags = find_target(target)

  times = [ check_up_to_date(inp) for inp in inputs ]

  myts = get_cache_time(target)
  if myts is None or any(ts > myts for ts in times):
    # Compute and cache a new value:
    ivalues = [ get_cached(inp)[1] for inp in inputs ]
    value = function(*ivalues)
    return cache_value(target, value, flags)
  else:
    # Just return time cached:
    return myts

def create(target):
  """
  Creates the desired target, using cached values when appropriate. Returns a
  (timestamp, value) pair indicating when the returned value was constructed.
  Raises a ValueError if the target is invalid or can't be created.
  """
  # Update dependencies as necessary (recursively)
  check_up_to_date(target)

  # Grab newly-cached value:
  ts, val = get_cached(target)

  # Double-check that we got a value:
  if val == NotAvailable:
    raise ValueError("Failed to create target '{}'.".format(target))

  return (ts, val)
