"""
utils.py

Miscellaneous utility functions.
"""

def default_params(defaults):
  """
  Returns a decorator which attaches the given dictionary as default parameters
  for the decorated function. Any keyword arguments supplied manually will
  override the provided defaults, and non-keyword arguments are passed through
  normally.
  """
  def wrap(function):
    def withargs(*args, **kwargs):
      merged = {}
      merged.update(defaults)
      merged.update(kwargs)
      return function(*args, **merged)
    return withargs
  return wrap

def reflow(src):
  """
  Reflows text so that only 2+ newlines count.
  """
  # get rid of extra newlines at front and back:
  src = re.sub("^\s*([^\n])", r"\1", src)
  src = re.sub("([^\n])\s*$", r"\1", src)
  # get rid of newlines between non-empty lines:
  src = re.sub("([^\n])[ \t]*\n[ \t]*([^\n])", r"\1 \2", src)
  # filter multiple newlines down to one:
  src = re.sub("\n\s*", "\n", src)
  return src
