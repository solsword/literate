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
import time

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

TABBR = ('y', 'w', 'd', 'h', 'm', 's')
def ftime(seconds):
  """
  Formats a time value in seconds into one of several human-readable formats.
  """
  if seconds > 60:
    seconds = int(seconds)
    mn, sc = divmod(seconds, 60)
    hr, mn = divmod(mn, 60)
    dy, hr = divmod(hr, 24)
    wk, dy = divmod(dy, 7)
    yr, wk = divmod(wk, 52)
    t = (yr, wk, dy, hr, mn, sc)
    return ''.join(
      "{:02d}{}".format(t[i], TABBR[i])
        for i in range(len(t))
          if any(x > 0 for x in t[:i+1])
    )
  else:
    return "{:.2g}s".format(seconds)

PR_N_EST = 8
PR_START = None
PR_INTRINSIC = 0
PR_ESTIMATES = [0]*PR_N_EST
PR_LAST = None
PR_LL = None
PR_CHARS = "▁▂▃▄▅▆▇█▇▆▅▄▃▂"
def prbar(progress, debug=print, interval=1, width=50):
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
  global PR_INTRINSIC, PR_START, PR_ESTIMATES, PR_LAST, PR_LL
  PR_INTRINSIC = (PR_INTRINSIC + 1) % len(PR_CHARS)
  if PR_INTRINSIC % interval != 0:
    return
  ic = PR_CHARS[PR_INTRINSIC]
  # estimate progress:
  now = time.monotonic()
  if PR_START == None:
    PR_START = now
  if PR_LAST == None:
    advanced = progress
    elapsed = 0
  else:
    advanced = progress - PR_LAST[0]
    elapsed = now - PR_LAST[1]
  if advanced != 0:
    est = (elapsed/advanced) * (1.0 - progress)
    PR_ESTIMATES = PR_ESTIMATES[1:] + [est]
    sm_est = sum(PR_ESTIMATES) / len(PR_ESTIMATES)
    estr = '~' + ftime(sm_est)
  else:
    if PR_LAST == None:
      estr = "-starting-"
    else:
      estr = "-stalled-"
  PR_LAST = (progress, now)
  # print bar:
  sofar = int(width * progress)
  left = width - sofar - 1
  if PR_LL != None:
    debug('\r' + (' '*PR_LL), end="")
  bar = "\r[{}>{}] ({}) {}".format("="*sofar, "-"*left, ic, estr)
  PR_LL = len(bar)-1
  debug(bar, end="")

def prdone(debug=print, width=50):
  """
  Call after a loop that uses prbar to finish off the progress bar, including
  printing the final time taken.
  """
  global PR_INTRINSIC, PR_START, PR_ESTIMATES, PR_LAST, PR_LL
  ttime = time.monotonic() - PR_START
  tstr = ftime(ttime)

  if PR_LL != None:
    debug('\r' + (' '*PR_LL), end="")
  debug("\r[{}>] ({}) {}".format("="*(width-1), '█', tstr))
  PR_INTRINSIC = 0
  PR_START = None
  PR_ESTIMATES = [0]*PR_N_EST
  PR_LAST = None
  PR_LL = None
