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
class LinearEq:
  '''
  m : float
    Slope of the line.
  y_int : float
    Y-intercept of the line.
  x_int : float
    X-intercept of the line.
  '''
  def __init__(self, m: float, y: float, x: float) -> None:
    self.m = m   
    self.y = y
    self.x = x

def antoine_T(v: npt.ArrayLike, P: npt.ArrayLike) -> npt.ArrayLike:
  '''
  Calculates the temperature of every component for each pressure.
  '''
  v = np.atleast_1d(v); P = np.atleast_1d(P)
  return (-v[:, 1] / (np.log10(P) - np.r_[v[:, 0]])) - v[:, 2]

def antoine_P(v: npt.ArrayLike, T: npt.ArrayLike) -> npt.ArrayLike:
  '''
  Calculates the pressure of every component for each temperature.
  '''
  v = np.atleast_1d(v); T = np.atleast_1d(T)
  return 10 ** (np.c_[v[:, 0]] - np.c_[v[:, 1]] / (T + np.c_[v[:, 2]]))

def lin_estimate_error(x_pair: list, y_pair: list) -> float:
  '''
  Calculates the x-intercept (x=0) for a given pair of x and y points. Assumes linearity.
  '''
  return x_pair[0] - y_pair[0] * ((x_pair[1]-x_pair[0])/(y_pair[1]-y_pair[0]))

def linear_intersect(line1: LinearEq, line2: LinearEq) -> tuple[float, float]:
  # TODO make docstrings!
  if line2.m == line1.m and line1.y != line2.y:
    return None, None
  x = (line1.y - line2.y)/(line2.m - line1.m)
  y = line1.m * x + line1.y
  return x, y
