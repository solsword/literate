"""
utils.py

Miscellaneous utility functions.

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
