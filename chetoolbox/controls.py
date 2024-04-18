import numpy as np
from numpy import typing as npt

def routh(coeff : npt.NDArray):
  coeff = np.array([.25, 1.25, 1.5625, 1.75])
  coeff = coeff.reshape(2, -1, order="F")
  det = np.zeros_like(coeff[0])
  det[:det.size-1] = (coeff[-1, :-1] * coeff[-2, 1:] - coeff[-2, :-1] * coeff[-1, 1:]) / coeff[-1, :-1]
  det

