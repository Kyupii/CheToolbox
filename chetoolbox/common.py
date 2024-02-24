import numpy as np
import numpy.typing as npt
from typing import Callable, Optional

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

class Equation:
  def __init__(self):
    return
  
  def eval(self, x: float) -> float:
    return x
  
  def inv(self, y: float) -> float:
    return y

class LinearEq(Equation):
  '''
  m : float
    Slope of the line.
  b : float
    Y-intercept of the line.
  x_int : float
    X-intercept of the line.
  eval : Callable
    Return the output of the function (y) when evaluated at an input (x).
  inv : Callable
    Return the input of the function (x) that evaluates to an output (y).
  '''
  def __init__(self, m: float, b: float, x_int: float | None = None) -> None:
    self.m = m
    self.b = b
    if np.isnan(self.m): # vertical lines
      self.m = np.NaN
      self.b = np.NaN
      self.x_int = x_int
    else:
      if b == 0.: # intersects the origin
        self.x_int = 0.
      else:
        if self.m == 0.: # horizontal lines
          self.x_int = np.NaN
        else:
          self.x_int = -m/b
  
  def eval(self, x: float | npt.ArrayLike) -> float | npt.ArrayLike: # numpy compatible
    if np.isnan(self.m): # vertical line
      return np.full_like(x, np.NaN)
    elif self.m == 0.: # horizontal line
      return np.full_like(x, self.b)
    else:
      return self.m * x + self.b
  
  def inv(self, y: float | npt.ArrayLike) -> float | npt.ArrayLike: # numpy compatible
    if np.isnan(self.m): # vertical line
      return np.full_like(y, self.x_int)
    elif self.m == 0.: # horizontal line
      return np.full_like(y, np.NaN)
    else:
      return (y - self.b) / self.m

class EqualibEq(Equation):
  '''
  alpha : float
    Equalibrium ratio (K1 / K2) between two species.
  eval : Callable
    Return the output of the function (y) when evaluated at an input (x).
  inv : Callable
    Return the input of the function (x) that evaluates to an output (y).
  '''
  def __init__(self, alpha: float) -> None:
    self.alpha = alpha
  
  def eval(self, x: float | npt.ArrayLike) -> float | npt.ArrayLike: # numpy compatible
    # breaks if x = -1. / (1. - self.alpha)
    return (self.alpha * x ) / (1. + (self.alpha - 1.) * x)
  
  def inv(self, y: float | npt.ArrayLike) -> float | npt.ArrayLike: # numpy compatible
    # breaks if y = -self.alpha / (1. - self.alpha)
    return y / (self.alpha + y * (1. - self.alpha))

class PiecewiseEq_new(Equation):
  '''
  Piecewise must be continuous (component equations must be equal at each bound) and injective (must have only one x per y value).

  eqs : tuple[Equation]
    All equations that compose the piecewise function. Equations must be ordered from smallest to largest upper domain limit.
  upperdomainlims : float
    The upper domain limit of each equation, except the last equation which has an upper domain limit of np.inf. Upperbounds must be ordered smallest to largest. Length must be len(eqs) -
  eval : Callable
    Return the output of the function (y) when evaluated at an input (x).
  inv : Callable
    Return the input of the function (x) that evaluates to an output (y).
  '''
  def __init__(self, eqs: tuple[Equation], upperdomainlims: tuple[float]):
    np.piecewise()
class PiecewiseEq(Equation):
  '''
  Piecewise must be continuous (component equations must be equal at each bound) and injective (must have only one x per y value).

  eqs : tuple[Equation]
    All equations that compose the piecewise function. Equations must be ordered from smallest to largest upper domain limit.
  upperdomainlims : float
    The upper domain limit of each equation, except the last equation which has an upper domain limit of np.inf. Upperbounds must be ordered smallest to largest. Length must be len(eqs) -
  eval : Callable
    Return the output of the function (y) when evaluated at an input (x).
  inv : Callable
    Return the input of the function (x) that evaluates to an output (y).
  '''
  def __init__(self, eqs: tuple[Equation], upperdomainlims: tuple[float]):
    if len(eqs) - 1 != len(upperdomainlims):
      raise AttributeError("Number of bounds do not match number of equations minus one.")
    upperdomainlims = np.array(list(upperdomainlims) + [np.inf])
    self.boundeqs = dict(zip(upperdomainlims.astype(str), eqs))
    self.posSlope = eqs[0].eval(upperdomainlims[0] - .05) < eqs[0].eval(upperdomainlims[0])

  def eval(self, x: float | npt.ArrayLike) -> float | npt.ArrayLike: # numpy compatible
    bounds = np.fromiter(self.boundeqs.keys(), float)
    print(type(x))
    if type(x) != npt.ArrayLike:
      eq = self.boundeqs[str(bounds[np.sum(bounds <= x)])]
      return eq.eval(x)
    else:
      splitind = np.less_equal(x, bounds[:, np.newaxis]).sum(axis=1)
      print(x, splitind)
      xsets = np.split(x, splitind)
      res = [self.boundeqs[bounds[i]].eval(xsets) for i in np.arange(len(xsets))]
      return np.concatenate(res)
  
  def inv(self, y: float | npt.ArrayLike) -> float | npt.ArrayLike: # NOT numpy compatible
    bounds = np.fromiter(self.boundeqs.keys(), float)
    curves = list(self.boundeqs.values())
    boundeqval = np.array([curve.eval(bounds[i]) for i, curve in enumerate(curves[:-1])])
    if self.posSlope:
      eq = self.boundeqs[str(bounds[np.sum(boundeqval <= y)])]
    else:
      eq = self.boundeqs[str(bounds[np.sum(boundeqval >= y)])]
    return eq.inv(y)

class SolutionObj(dict):
  def __getattr__(self, name):
    try: 
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e

  def __dir__(self) -> tuple:
    return tuple(self.keys())
  
  def unpack(self) -> tuple:
    return tuple(self.values())

class UnitConv:
  def faren2kelvin(T):
    T = np.atleast_1d(T)
    return (T - 32.) * (5./9.) + 273.15
  def ft2meters(ft):
    ft = np.atleast_1d(ft)
    return ft * .3048
  def lbs2kgs(lbs):
    lbs = np.atleast_1d(lbs)
    return lbs * .4535934

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

def raoult_XtoY(x: list, K: list) -> tuple[npt.ArrayLike, float]:
  '''
  Calculates the vapor mole fraction of a multi-component mixed phase feed (assuming liquid and gas ideality).

  Parameters
  ---------
  x : list
    Component mole fractions of the feed's liquid phase (unitless). Must sum to 1.
  K : list 
    Equalibrium constant for each component at a specific temperature and pressure (units vary). Length must equal x.
  
  Returns
  ---------
  y : ArrayLike
    Component mole fractions of the feed's vapor phase (unitless).
  error : float
    Error of calculated vapor phase component mole fractions.
  '''
  x = np.atleast_1d(x)
  K = np.atleast_1d(K)
  y = np.c_[x] * K
  error = np.sum(y) - 1
  return y, error

def raoult_YtoX(y: list, K: list) -> tuple[npt.ArrayLike, float]:
  '''
  Calculates the liquid mole fraction of a multi-component mixed phase feed (assuming liquid and gas ideality).

  Parameters
  ---------
  y : list
    Component mole fractions of the feed's vapor phase (unitless). Must sum to 1.
  K : list 
    Equalibrium constant for each component at a specific temperature and pressure (units vary). Length must equal y.
    
  Returns
  ---------
  x : ArrayLike
    Component mole fractions of the feed's liquid phase (unitless).
  error : float
    Error of calculated liquid phase component mole fractions.
  '''
  y = np.atleast_1d(y)
  K = np.atleast_1d(K)
  x = np.c_[y] / K
  error = np.sum(x) - 1
  return x, error

def point_separsort(*points: list | tuple | npt.ArrayLike) -> tuple[npt.ArrayLike, npt.ArrayLike]:
  '''
  Takes an arbitrary set of points and sorts them by x-value. Returns them as two np.arrays, one of x-values and the other of y-values.
  '''
  points = np.atleast_1d(points).reshape(-1, 2)
  points = points[points[:, 0].argsort()]
  return points[:, 0], points[:, 1]

def array_pad(arrs: tuple[npt.ArrayLike]) -> npt.ArrayLike:
  '''
  Takes an arbitrary list of np.arrays of varying length and pads each np.array to form a homogeneous 2D array of size len(arrs) x max(lens)
  '''
  lens = np.array([len(v) for v in arrs])
  mask = lens[:, None] > np.arange(lens.max())
  out = np.full(mask.shape, np.NaN)
  out[mask] = np.concatenate(arrs)
  return out

def lin_estimate_error(x_pair: npt.ArrayLike, y_pair: npt.ArrayLike) -> float:
  '''
  Calculates the x-intercept (x=0) for a given pair of x and y distances. Assumes linearity.
  '''
  x_pair = np.atleast_1d(x_pair); y_pair = np.atleast_1d(y_pair)
  x_new = x_pair[0] - y_pair[0] * ((x_pair[1]-x_pair[0])/(y_pair[1]-y_pair[0]))
  return x_new

def err_reduc(err_calc: Callable[[float], float], x: npt.ArrayLike) -> tuple[npt.ArrayLike, npt.ArrayLike]:
  '''
  Evaluates an error calculation for a pair of inputs and returns a new set of inputs with a smaller average error.
  '''
  x = np.atleast_1d(x)
  err = err_calc(x)
  xnew = lin_estimate_error(x, err)
  err = np.abs(err)
  x[np.argmax(err)] = xnew
  return x, err

def iter(err_calc: Callable[[float], float], x: npt.ArrayLike, tol: float = .001) -> tuple[float, float, int]:
  '''
  Accepts a pair of inputs and an error function. Returns an input with tolerable error, the error, and the iterations required.
  '''
  x = np.atleast_1d(x)
  error = 10000.
  i = 0
  while np.min(error) > tol:
    x, error = err_reduc(err_calc, x)
    i += 1
  return x[np.argmin(error)], np.min(error), i

def vertical_line(x) -> LinearEq:
  return LinearEq(np.NaN, np.NaN, x)

def horizontal_line(y) -> LinearEq:
  return LinearEq(0., y)

def point_conn(point1: tuple[float, float], point2: tuple[float, float]) -> LinearEq:
  '''
  Calculates equation of a line from two points.
  '''
  point1 = np.atleast_1d(point1); point2 = np.atleast_1d(point2)
  if point1[0] == point2[0]:
    return vertical_line(point1[0])
  elif point1[1] == point2[1]:
    return horizontal_line(point1[1])
  m = (point1[1] - point2[1]) / (point1[0] - point2[0])
  b = point1[1] - m * point1[0]
  return LinearEq(m, b)

def point_slope(point: tuple[float, float], slope: float ) -> LinearEq:
  '''
  Calculates equation of a line from a point and its slope.
  '''
  point = np.atleast_1d(point)
  if np.isnan(slope):
    return vertical_line(point[0])
  elif slope == 0.:
    return horizontal_line(point[1])
  else:
    return LinearEq(slope, -slope * point[0] + point[1])

def linear_intersect(line1: LinearEq, line2: LinearEq) -> tuple[float, float] | None:
  '''
  Calculates the intersection points of two straight lines or returns None if no intersect exists. Uses LinearEq objects.
  '''
  if line2.m == line1.m and line1.b != line2.b:
    return None
  if np.isnan(line1.m) or np.isnan(line2.m):
    if np.isnan(line1.m):
      return line1.x_int, line2.eval(line1.x_int)
    else:
      return line2.x_int, line1.eval(line2.x_int)
  else:
    x = (line1.b - line2.b)/(line2.m - line1.m)
  return x, line1.eval(x)

def quadratic_formula(coeff: npt.ArrayLike) -> npt.ArrayLike | None:
  '''
  Calculates the roots of a quadratic equation. Ignores imaginary roots.
  '''
  coeff = np.atleast_1d(coeff)
  descrim = coeff[1]**2 - 4*coeff[0]*coeff[2]
  if descrim < 0:
    return None
  return (- coeff[1] + np.sqrt(np.array([descrim])) * np.array([1, -1])) / (2. * coeff[0])

def curve_bouncer(upper: Equation, lower: Equation, x_start: float, x_stop: float, x_transform: Callable[[float, float], float] | None = None, y_transform: Callable[[float, float], float] | None = None) -> float:
  '''
  Bounce between two curves and return the number of bounces required to reach x_stop. x_start must cross both curves. If x_transform is None, move straight vertically. If y_transform is None, move straight horizontal. Each transform function must accept only its own variable and return only its own variable.
  '''
  x = x_start
  y = upper.eval(x_start)
  i = 0
  while x > x_stop:
    xprev = x
    if x_transform != None:
      x = x_transform(x)
    y = lower.eval(x)
    if y_transform != None:
      y = y_transform(y)
    x = upper.inv(y)
    i += 1
  return (i - 1.) + (xprev - x_stop) / (xprev - x)