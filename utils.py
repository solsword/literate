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

import re

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

PR_INTRINSIC = 0
PR_CHARS = "▁▂▃▄▅▆▇█▇▆▅▄▃▂"
def prbar(progress, debug=print, interval=1, width=65):
  """
  Prints a progress bar. The argument should be a number between 0 and 1. Put
  this in a loop without any other printing and the bar will fill up on a
  single line. To print stuff afterwards, use an empty print statement after
  the end of the loop to move off of the progress bar line.

  The output will be sent to the given 'debug' function, which is just "print"
  by default.
  
  If an 'interval' value greater than 1 is given, the bar will only be printed
  every interval calls to the function.
  
  'width' may also be specified to determine the width of the bar in characters
  (but note that 6 extra characters are printed, so the actual line width will
  be width + 6).
  """
  global PR_INTRINSIC
  ic = PR_CHARS[PR_INTRINSIC]
  PR_INTRINSIC = (PR_INTRINSIC + 1) % len(PR_CHARS)
  if PR_INTRINSIC % interval != 0:
    return
  sofar = int(width * progress)
  left = width - sofar - 1
  debug("\r[" + "="*sofar + ">" + "-"*left + "] (" + ic + ")", end="")
