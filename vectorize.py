"""
vectorize.py

Code for converting characters into vectors (and back).
"""

STX = '\u0002'
ETX = '\u0003'
SUB = '\u001a'

MAX_UNICHR = 0x10ffff
PRACTICAL_MAX_UNICHR = 0x2ffff

import numpy as np

def univec(c):
  """
  Translates a character into a binary vector starting from the Unicode
  codepoint and concatenating the one-hot encodings of each base-32 digit. Only
  allows up to 4 base-32 digits, which is enough to cover every codepoint in
  the basic multilingual plan, the supplementary multilingual plane, and the
  supplementary ideographic plane. Characters outside this range are replaced
  by the substitute character.
  """
  result = np.array([0]*(32*4), dtype=bool)
  o = ord(c)
  if o > PRACTICAL_MAX_UNICHR:
    o = ord(SUB)
  # unrolled loop x4; 32**5 > 0x10ffff
  result[32*0 + (o % 32)] = 1
  o //= 32
  result[32*1 + (o % 32)] = 1
  o //= 32
  result[32*2 + (o % 32)] = 1
  o //= 32
  result[32*3 + (o % 32)] = 1
  o //= 32
  return result

def unichr(v):
  """
  Translates a binary vector into a character. Uses argmax on each 32-entry
  chunk, so works even if the vector isn't purely binary.
  """
  rord = 0
  denom = 1
  for i in range(4):
    rord += denom * np.argmax(v[:32])
    v = v[32:]
    denom *= 32

  if rord > MAX_UNICHR:
    return SUB
  else:
    return chr(rord)


def vectorize(params, text, pad=False):
  """
  Turns some text into a bit-string encoded version, returning the encoded text
  as a two-dimensional numpy array where each row encodes a character from the
  text and each column corresponds to a character. UTF-32 codepoint bits are
  used as the encoding.

  If pad is given as True, encoded STX and ETX will be used to pad the string
  by window_size on either side.
  """
  padding = params["window_size"] if pad else 0
  
  return np.array(
    [univec(STX)] * padding + [
      univec(c) for c in text
    ] + [univec(ETX)] * padding,
    dtype=np.bool
  )

def de_vectorize(params, vec):
  """
  Converts a bit-array encoding as returned by vectorize back into characters
  using the given character map. Uses a threshold of 0.5 in case the input
  isn't really binary.
  """
  result = ''
  for row in vec:
    result += unichr(row)

  return result
