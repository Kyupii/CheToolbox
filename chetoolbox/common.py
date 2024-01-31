import numpy as np
import numpy.typing as npt

class TESTVARS:
  '''
  <Function/Class Description>.

  Parameters:
  -----------
  A : <Variable Type>
    <Variable Description> in <Unit Abbreviation/Symbols> (<Unit In Spoken English>).
  B : <Variable Type>
    <Variable Description>.
      Ex) <Optional Example of Variable Shape, Formatting, Etc.>.
  
  Retruns:
  -----------
  C : <Variable Type>
    <Variable Description> in <Unit Abbreviation/Symbols> (<Unit In Spoken English>).
  '''
  def __init__(self) -> None:
    pass
  numb_comp = 3
  mol_frac = np.array([.25, .40, .35])
  x = np.array([.25, .40, .35])
  y = np.array([0.576857, 0.312705, 0.102059])
  ant_coeff = np.array([[6.82973, 813.2, 248.],
                        [6.83029, 945.90, 240.],
                        [6.85221, 1064.63, 232.]])
  P = 10342.95

def antoine_T(v: npt.ArrayLike, P: npt.ArrayLike) -> npt.ArrayLike:
  '''
  Calculates the temperature of every component for each pressure.
  '''
  v = np.atleast_1d(v); P = np.atleast_1d(P)

  def antoine_T_3(v, P):
    return (-v[:, 1] / (np.log10(P) - np.r_[v[:, 0]])) - v[:, 2]
  
  if len(v[1, :]) == 3:
    return antoine_T_3(v, P)
  else:
    T = np.zeros((len(v), len(P)))
    for i_comp, ant_coeff in enumerate(v):
      for j_temp, press in enumerate(P):
        a, b, c = ant_coeff
        T[i_comp, j_temp] = -b / (np.log10(press) - a) - c
    return T

def antoine_P(v: npt.ArrayLike, T: npt.ArrayLike) -> npt.ArrayLike:
  '''
  Calculates the pressure of every component for each temperature.
  '''
  v = np.atleast_1d(v); T = np.atleast_1d(T)

  def antoine_P_3(v, T):
    return 10 ** (np.c_[v[:, 0]] - np.c_[v[:, 1]] / (T + np.c_[v[:, 2]]))
  
  if len(v[1, :]) == 3:
    return antoine_P_3(v, T)
  else:
    P = np.zeros((len(v), len(T)))
    for i_comp, ant_coeff in enumerate(v):
      for j_temp, temp in enumerate(T):
        a, b, c = ant_coeff
        P[i_comp, j_temp] = 10 ** (a - (b) / (temp + c))
    return P

def lin_estimate_error(x_pair: list, y_pair: list) -> float:
  '''
  Calculates the x-intercept (x=0) for a given pair of x and y points. Assumes linearity.
  '''
  return x_pair[0] - y_pair[0] * ((x_pair[1]-x_pair[0])/(y_pair[1]-y_pair[0]))
