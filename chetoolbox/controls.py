import numpy as np
from numpy import typing as npt
from . import common

def routh(coeff : npt.NDArray):
  coeff = np.atleast_1d(coeff)
  if coeff.size % 2 != 0:
    coeff = np.append(coeff, 0.)
  coeff = coeff.reshape(2, -1, order="F")
  det = np.vstack((coeff, np.zeros((coeff[0].size, coeff[0].size))))
  for i in np.arange(2, 2 + det[0].size):
    pos = det[i-1, :-i//2] * det[i-2, 1:det[0].size + (-i//2) + 1]
    neg = det[i-2, :-i//2] * det[i-1, 1:det[0].size + (-i//2) + 1]
    det[i, :-i//2] = (pos - neg) / det[i-1, :-i//2]
    # pretty printing !!
    # print(f"{det[i-1, 0]} * {det[i-2, 1]} - {det[i-2, 0]} * {det[i-1, 1]} / {det[i-1, 0]}")
    # print(det[i, 0])
    # print(f"{det[i-1, 1]} * {det[i-2, 2]} - {det[i-2, 1]} * {det[i-1, 2]} / {det[i-1, 1]}")
    # print(det[i, 1])
  return common.SolutionObj(det = det, stable = np.sign(det[:, 0]).sum() in {-det[:, 0].size, det[:, 0].size})
