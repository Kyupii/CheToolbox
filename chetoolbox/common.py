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
  b : float
    Y-intercept of the line.
  x_int : float
    X-intercept of the line.
  eval : function
    Return the output of the function (y) when evaluated at an input (x).
  inv : function
    Return the input of the function (x) that evaluates to an output (y).
  '''
  def __init__(self, m: float, b: float) -> None:
    self.m = m   
    self.b = b
    self.x_int = -m/b
  
  def eval(self, x: float) -> float: # numpy compatible
    return self.m * x + self.b
  
  def inv(self, y: float) -> float: # numpy compatible
    return (y - self.b) / self.m
  
class EqualibEq:
  '''
  alpha : float
    Equalibrium ratio (K1 / K2) between two species.
  eval : function
    Return the output of the function (y) when evaluated at an input (x).
  inv : function
    Return the input of the function (x) that evaluates to an output (y).
  '''
  def __init__(self, alpha: float) -> None:
    self.alpha = alpha
  
  def eval(self, x: float) -> float: # numpy compatible
    # breaks if x = -1. / (1. - self.alpha)
    return self.alpha * x / (1. + (1. - self.alpha) * x)
  
  def inv(self, y: float) -> float: # numpy compatible
    # breaks if y = -self.alpha / (1. - self.alpha)
    return y / (self.alpha + y * (1. - self.alpha))

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

def lin_estimate_error(x_pair: npt.ArrayLike, y_pair: npt.ArrayLike) -> float:
  '''
  Calculates the x-intercept (x=0) for a given pair of x and y distances. Assumes linearity.
  '''
  x_pair = np.atleast_1d(x_pair); y_pair = np.atleast_1d(y_pair)
  return x_pair[0] - y_pair[0] * ((x_pair[1]-x_pair[0])/(y_pair[1]-y_pair[0]))

def err_reduc(err_calc: function, x: npt.ArrayLike) -> tuple[npt.ArrayLike, npt.ArrayLike]:
  '''
  Evaluates an error calculation for a pair of inputs and returns a new set of inputs with a smaller average error.
  '''
  x = np.atleast_1d(x)
  err = err_calc(x)
  xnew = lin_estimate_error(x, err)
  err = np.abs(err)
  x[np.argmin(err)] = xnew
  return x, err

def iter(err_calc: function, x: npt.ArrayLike, tol: float = .001) -> tuple[float, float, int]:
  '''
  Accepts a pair of inputs and an error function. Returns an input with tolerable error, the error, and the iterations required.
  '''
  x = np.atleast_1d(x)
  error = 10000.
  i = 0
  while np.min(error) > tol:
    error, x = err_reduc(err_calc, x)
    i += 1
  return x[np.argmin(error)], np.min(error), i

def vertical_line(x) -> LinearEq:
  line = LinearEq(0, 1)
  line.m = np.NaN
  line.b = np.NaN
  line.x_int = x
  return line

def point_slope(point1: tuple, point2: tuple) -> LinearEq:
  '''
  Calculates equation of a line from two points.
  '''
  point1 = np.atleast_1d(point1); point2 = np.atleast_1d(point2)
  if point1[0] == point2[0]:
    return vertical_line(point1[0])
  m = (point1[1]-point2[1])/(point1[0]-point2[0])
  b = point1[1] - m * point1[0]
  return LinearEq(m, b)
  
def linear_intersect(line1: LinearEq, line2: LinearEq) -> tuple[float, float] | None:
  '''
  Calculates the intersection points of two straight lines or None if no intersect exists. Uses LinearEq objects.
  '''
  if line2.m == line1.m and line1.y != line2.y:
    return None
  x = (line1.b - line2.b)/(line2.m - line1.m)
  y = line1.eval(x)
  return x, y

def quadratic_formula(coeff: npt.ArrayLike) -> npt.ArrayLike | None:
  '''
  Calculates the roots of a quadratic equation. Ignores imaginary roots.
  '''
  coeff = np.atleast_1d(coeff)
  descrim = coeff[1]**2 - 4*coeff[0]*coeff[2]
  if descrim < 0:
    return None
  return (np.sqrt(np.array([descrim])).repeat(2) * np.array([1, -1]) - coeff[1]) / (2. * coeff[0])