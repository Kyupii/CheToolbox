import numpy as np
import numpy.typing as npt
from typing import Callable

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
  
  def eval(self, x: float | npt.NDArray) -> float | npt.NDArray:
    return x
  
  def inv(self, y: float | npt.NDArray) -> float | npt.NDArray:
    return y
  
  def deriv(self, x: float | npt.NDArray) -> float | npt.NDArray:
    return x
  
  def integ(self, x1: float | npt.NDArray, x2: float | npt.NDArray) -> float | npt.NDArray:
    return x2 - x1

class LinearEq(Equation):
  '''
  Equation of the form y = m*x + b
  -----------
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
  deriv : Callable
    Return the derivative of the function at an input (x).
  inv : Callable
    Return the integral (area under the curve) of a function between inputs (x1 and x2). If both x1 and x2 are np.arrays then size must match.
  '''
  def __init__(self, m: float, b: float, x_int: float | None = None):
    self.m = m
    self.b = b
    if np.isnan(self.m): # vertical lines
      self.x_int = x_int
      if self.x_int == 0.:
        self.b = 0.
      else:
        self.b = np.NaN
    else:
      if b == 0.: # intersects the origin
        self.x_int = 0.
      else:
        if self.m == 0.: # horizontal lines
          self.x_int = np.NaN
        else:
          self.x_int = -b/m
  
  def eval(self, x: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    return self.m * x + self.b
  
  def inv(self, y: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    if self.m == 0.: # horizontal line
      return y * np.NaN
    else:
      return (y - self.b) / self.m
  
  def deriv(self, x: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    return x * 0 + self.m
  
  def integ(self, x1: float | npt.NDArray, x2: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    '''
    x1 and x2 must be the same size if both are arrays.
    '''
    return .5 * self.m * (x2**2 - x1**2) + self.b * (x2 - x1)

class QuadraticEq(Equation):
  '''
  Equation of the form y = a*(x - s1)**2 + b*(x - s2) + c
  -----------
  a : float
    Coefficient of the quadratic term
  b : float
    Coefficient of the linear term
  c : float
    Y-intercept of the curve.
  s1 : float
    Curve shift in the quadratic term.
  s2 : float
    Curve shift in the linear term.
  eval : Callable
    Return the output of the function (y) when evaluated at an input (x).
  inv : Callable
    Return the input of the function (x) that evaluates to an output (y).
  deriv : Callable
    Return the derivative of the function at an input (x).
  inv : Callable
    Return the integral (area under the curve) of a function between inputs (x1 and x2). If both x1 and x2 are np.arrays then size must match.
  '''
  def __init__(self, a: float, b: float, c: float, s1: float = 0., s2: float = 0.):
    self.a = a
    self.b = 2. * a * s1 + b
    self.c = a * s1**2. - b * s2 + c
    self.determ = self.b**2 - 4.*self.a*self.c
    self.roots = quadratic_formula([self.a, self.b, self.c])
  
  def eval(self, x: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    return self.a*x**2 + self.b*x + self.c
  
  def inv(self, y: float | npt.NDArray, mode: str = "both") -> float | npt.NDArray | None: # numpy compatible
    '''
    mode must be "both", "left", or "right"
    '''
    leftailias = {"left", "l"}; rightalias = {"right", "r"}
    w = self.b / (self.a * 2.)
    z = self.c / self.a - w
    revdescrim = y - z
    if revdescrim < 0.:
      return None
    if mode in leftailias:
      return -np.sqrt(revdescrim) - w
    elif mode in rightalias:
      return np.sqrt(revdescrim) - w
    else:
      return np.array([1., -1.]) * np.sqrt(revdescrim) - w
  
  def deriv(self, x: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    return 2. * self.a * x + self.b
  
  def integ(self, x1: float | npt.NDArray, x2: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    '''
    x1 and x2 must be the same size if both are arrays.
    '''
    return (1./3.) * self.m * (x2**3 - x1**3) + .5 * self.b * (x2**2 - x1**2) + self.c * (x2 - x1)

class CubicEq(Equation):
  '''
  Equation of the form y = a*(x - s1)**3 + b*(x - s2)**2 + c*(x - s3) + d
  -----------
  a : float
    Coefficient of the cubic term
  b : float
    Coefficient of the quadratic term
  c : float
    Coefficient of the linear term
  d : float
    Y-intercept of the curve.
  s1 : float
    Curve shift in the cubic term.
  s2 : float
    Curve shift in the quadratic term.
  s3 : float
    Curve shift in the linear term.
  eval : Callable
    Return the output of the function (y) when evaluated at an input (x).
  inv : Callable
    Return the input of the function (x) that evaluates to an output (y).
  deriv : Callable
    Return the derivative of the function at an input (x).
  inv : Callable
    Return the integral (area under the curve) of a function between inputs (x1 and x2). If both x1 and x2 are np.arrays then size must match.
  '''
  def __init__(self, a: float, b: float, c: float, d: float, s1: float = 0., s2: float = 0., s3: float = 0.):
    self.a = a
    self.b = b - 3.*a*s1
    self.c = 3.*a*s1**2 + c - 2.*b*s2
    self.d = d - a*s1**3 + b*s2**2 - c*s3
    self.determ = 18.*self.a*self.b*self.c*self.d - 4.*self.d*self.b**3 + self.c**2*self.b**2 - 4.*self.a*self.c - 27.*self.a**2*self.d**2
    self.roots = cubic_formula([self.a, self.b, self.c, self.d])
  
  def eval(self, x: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    return self.a*x**3 + self.b*x**2 + self.c*x + self.d
  
  def inv(self, y: float | npt.NDArray) -> float | npt.NDArray | None:
    # can be rewritten as (w*x + n)**3 + z (triple root case) iff b**2 == 3*c*a
    if self.b**2 == 3.*self.a*self.c:
      w = np.cbrt(self.a)
      n = self.b / (3.*w**2)
      z = self.d - n**3
      return (n + np.cbrt(y - z)) / w
    
    yfloat = type(y) not in {list, np.ndarray}; y = np.atleast_1d(y)
    locext = quadratic_formula([3.*self.a, 2.*self.b, self.c])
    if locext is None or len(locext) < 2:
      def error(x):
        return self.eval(x) - np.c_[y]
      
      starts = np.zeros((len(y), 2))
      starts[:, 0] = np.cbrt(y)
      starts[:, 1] = 1-np.cbrt(y)
      x, _, _ = err_reduc_iterative(error, starts)
      return x
      
    else: # not invertible, maybe implement piecewise-like method?
      raise ArithmeticError("Cubic equation is not differentiable")

  def deriv(self, x: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    return 3.*self.a*x**2 + 2.*self.b*x + self.c
  
  def deriv2(self, x: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    return 6.*self.a*x + 2.*self.b
  
  def integ(self, x1: float | npt.NDArray, x2: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    '''
    x1 and x2 must be the same size if both are arrays.
    '''
    return (1./4.)*self.a*(x2**4 - x1**4) + (1./3.)*self.b*(x2**3 - x1**3) + (1./2.)*self.c*(x2**2 - x1**2) + self.d*(x2 - x1)

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
  
  def eval(self, x: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    # breaks if x = -1. / (1. - self.alpha)
    x = np.minimum(np.ones_like(x), np.maximum(np.zeros_like(x), x))
    return (self.alpha * x ) / (1. + (self.alpha - 1.) * x)
  
  def inv(self, y: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    # breaks if y = -self.alpha / (1. - self.alpha)
    y = np.minimum(np.ones_like(y), np.maximum(np.zeros_like(y), y))
    return y / (self.alpha + y * (1. - self.alpha))

class PiecewiseEq(Equation):
  '''
  Piecewise must be continuous (component equations must be equal at each bound) and injective (must have only one x per y value).
  -----------
  eqs : tuple[Equation]
    All equations that compose the piecewise function. Equations must be ordered from smallest to largest upper domain limit.
  upperdomainlims : float
    The upper domain limit of each equation, except the last equation which has an upper domain limit of np.inf. Upperbounds must be ordered smallest to largest. Length must be len(eqs) - 1.
  eval : Callable
    Return the output of the function (y) when evaluated at an input (x).
  inv : Callable
    Return the input of the function (x) that evaluates to an output (y).
  deriv : Callable
    Return the derivative of the function at an input (x).
  inv : Callable
    Return the integral (area under the curve) of a function between inputs (x1 and x2). If both x1 and x2 are np.arrays then size must match.
  '''
  def __init__(self, eqs: tuple[Equation], upperdomainlims: tuple[float]):
    if len(eqs) - 1 != len(upperdomainlims):
      raise AttributeError("Number of bounds do not match number of equations minus one.")
    self.bounds = np.array(list(upperdomainlims) + [np.inf])
    self.eqs = np.array(eqs)
    self.boundeqs = dict(zip(self.bounds.astype(str), eqs))
    self.posSlope = eqs[0].eval(self.bounds[0] - .05) < eqs[0].eval(self.bounds[0])

  def eval(self, x: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    xfloat = type(x) not in {list, np.ndarray}; x = np.c_[np.atleast_1d(x)]
    eq_index = (x > self.bounds).sum(axis=1)
    ind_eq_x = np.hstack([np.c_[np.arange(len(x))], np.c_[eq_index], x])
    ind_eq_x = ind_eq_x[ind_eq_x[:, 1].argsort()]
    xs_per_eq = np.split(ind_eq_x, np.where(ind_eq_x[:, 1][:-1] != ind_eq_x[:, 1][1:])[0] + 1)
    res = np.vstack([np.hstack([np.c_[xset[:, 0]], np.c_[self.eqs[xset[0, 1].astype(int)].eval(xset[:, 2])]]) for xset in xs_per_eq])
    return res[0, 1] if xfloat else res[res[:, 0].argsort()][:, 1]

  def inv(self, y: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    yfloat = type(y) not in {list, np.ndarray}; y = np.c_[np.atleast_1d(y)]
    boundval = np.array([eq.eval(self.bounds[i]) for i, eq in enumerate(self.eqs)])
    eq_index = (y > boundval).sum(axis=1) if self.posSlope else (y < boundval).sum(axis=1)
    ind_eq_y = np.hstack([np.c_[np.arange(len(y))], np.c_[eq_index], y])
    ind_eq_y = ind_eq_y[ind_eq_y[:, 1].argsort()]
    ys_per_eq = np.split(ind_eq_y, np.where(ind_eq_y[:, 1][:-1] != ind_eq_y[:, 1][1:])[0] + 1)
    res = np.vstack([np.hstack([np.c_[yset[:, 0]], np.c_[self.eqs[yset[0, 1].astype(int)].inv(yset[:, 2])]]) for yset in ys_per_eq])
    return res[0, 1] if yfloat else res[res[:, 0].argsort()][:, 1]

  def deriv(self, x: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    xfloat = type(x) not in {list, np.ndarray}; x = np.c_[np.atleast_1d(x)]
    eq_index = (x > self.bounds).sum(axis=1)
    ind_eq_x = np.hstack([np.c_[np.arange(len(x))], np.c_[eq_index], x])
    ind_eq_x = ind_eq_x[ind_eq_x[:, 1].argsort()]
    xs_per_eq = np.split(ind_eq_x, np.where(ind_eq_x[:, 1][:-1] != ind_eq_x[:, 1][1:])[0] + 1)
    res = np.vstack([np.hstack([np.c_[xset[:, 0]], np.c_[self.eqs[xset[0, 1].astype(int)].deriv(xset[:, 2])]]) for xset in xs_per_eq])
    return res[0, 1] if xfloat else res[res[:, 0].argsort()][:, 1]

  def integ(self, x1: float | npt.NDArray, x2: float | npt.NDArray) -> float | npt.NDArray: # numpy compatible
    dualfloat = (type(x1) not in {list, np.ndarray}) & (type(x2) not in {list, np.ndarray})
    typemix = type(x1) != type(x2)
    x1 = np.c_[np.atleast_1d(x1)]; x2 = np.c_[np.atleast_1d(x2)]
    if typemix:
      if len(x1) > len(x2):
        x2 = np.full_like(x1, x2)
      else:
        x1 = np.full_like(x2, x1)
    eq_index = ((x1 > self.bounds) ^ (x2 > self.bounds)) # XOR both bounds

    # within one equation piece
    singles_index = np.arange(len(x1))[eq_index.sum(axis=1) == 0]
    singles = np.zeros((1,4))
    if len(singles_index) != 0:
      singles_eq_ind = (np.c_[x1[eq_index.sum(axis=1) == 0]] > self.bounds).sum(axis=1)
      singles = np.hstack([np.c_[singles_index], np.c_[singles_eq_ind], np.c_[x1[singles_index]], np.c_[x2[singles_index]]])

    # split across multiple equation pieces
    splits_index = np.arange(len(x1))[eq_index.sum(axis=1) != 0]
    splits = np.zeros((1,4))
    if len(splits_index) != 0:
      eq_index[np.where(eq_index[:, :-1] > eq_index[:, 1:])[0], np.where(eq_index[:, :-1] > eq_index[:, 1:])[1] + 1] = True
      regions = [np.concatenate([x1[splits_index[i]], self.bounds[ind][:-1], x2[splits_index[i]]]) for i, ind in enumerate(eq_index[eq_index.sum(axis=1) != 0])]
      boundpairs = np.vstack([np.lib.stride_tricks.sliding_window_view(reg, 2) for reg in regions])
      splits = np.hstack( (np.transpose(np.where(eq_index == True)), boundpairs))

    callstack = np.vstack((singles, splits))
    callstack = callstack[callstack[:, 1].argsort()]
    xs_per_eq = np.split(callstack, np.where(callstack[:, 1][:-1] != callstack[:, 1][1:])[0] + 1)
    res_per_eq = np.vstack([np.hstack([np.c_[xset[:, 0]], np.c_[self.eqs[xset[0, 1].astype(int)].integ(xset[:, 2], xset[:, 3])]]) for xset in xs_per_eq])
    res = np.hstack([np.sum(res_per_eq[:, 1], where=(res_per_eq[:, 0] == i)) for i in np.arange(len(x1))])
    return res[0] if dualfloat else res

class SolutionObj(dict):
  def __getattr__(self, name):
    try: 
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e
    
  def __repr__(self):
    if self.keys():
        return _dict_formatter(self)
    else:
        return self.__class__.__name__ + "()"
    
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

# region String Operations
def _dict_formatter(d, n = 0, mplus = 1, sorter = None):
  '''
  Pretty printer for dictionaries

  `n` keeps track of the starting indentation;
  lines are indented by this much after a line break.
  `mplus` is additional left padding applied to keys
  '''
  if isinstance(d, dict):
    m = max(map(len, list(d.keys()))) + mplus  # width to print keys
    s = '\n'.join([k.rjust(m) + ': ' +  # right justified, width m
    _indenter(_dict_formatter(v, m+n+2, 0, sorter), m+2)
    for k,v in d.items()])  # +2 for ': '
  else:
    # By default, NumPy arrays print with linewidth=76. `n` is
    # the indent at which a line begins printing, so it is subtracted
    # from the default to avoid exceeding 76 characters total.
    # `edgeitems` is the number of elements to include before and after
    # ellipses when arrays are not shown in full.
    # `threshold` is the maximum number of elements for which an
    # array is shown in full.
    # These values tend to work well for use with OptimizeResult.
    with np.printoptions(linewidth=76-n, edgeitems=2, threshold=12, formatter={'float_kind': _float_formatter_10}):
      s = str(d)
  return s

def _indenter(s, n = 0):
  '''
  Ensures that lines after the first are indented by the specified amount
  '''
  split = s.split("\n")
  indent = " "*n
  return ("\n" + indent).join(split)

def _float_formatter_10(x):
  '''
  Returns a string representation of a float with exactly ten characters
  '''
  if np.isposinf(x):
    return "       inf"
  elif np.isneginf(x):
    return "      -inf"
  elif np.isnan(x):
    return "       nan"
  return np.format_float_scientific(x, precision=3, pad_left=2, unique=False)
# endregion

# region Array Operations
def array_pad(arrs: tuple[npt.NDArray]) -> npt.NDArray:
  '''
  Takes an arbitrary list of np.arrays of varying length and pads each np.array to form a homogeneous 2D array of size len(arrs) x max(lens)
  '''
  lens = np.array([len(v) for v in arrs])
  mask = lens[:, None] > np.arange(lens.max())
  out = np.full(mask.shape, np.NaN)
  out[mask] = np.concatenate(arrs)
  return out

def point_separsort(*points: list | tuple | npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
  '''
  Takes an arbitrary set of points and sorts them by x-value. Returns them as two np.arrays, one of x-values and the other of y-values.
  '''
  points = np.atleast_1d(points).reshape(-1, 2)
  points = points[points[:, 0].argsort()]
  return points[:, 0], points[:, 1]
# endregion

# region Geometry
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

def quadratic_formula(coeff: npt.NDArray) -> npt.NDArray | None:
  '''
  Calculates the roots of a quadratic equation. Ignores imaginary roots.
  '''
  coeff = np.atleast_1d(coeff)
  descrim = coeff[1]**2 - 4.*coeff[0]*coeff[2]
  if descrim < 0.:
    return None
  return (- coeff[1] + np.sqrt(descrim) * np.array([1., -1.])) / (2. * coeff[0])

def cubic_formula(coeff: npt.NDArray) -> npt.NDArray:
  '''
  Calculates the roots of a cubic equation. Ignores imaginary roots.
  '''
  coeff = np.atleast_1d(coeff)
  descrim = 18.*np.prod(coeff) - 4.*coeff[3]*coeff[1]**3 + coeff[2]**2*coeff[1]**2 - 4.*coeff[0]*coeff[2] - 27.*coeff[0]**2*coeff[3]**2
  if descrim == 0.: # multiple root present!
    trip = coeff[1]**2 - 3.*coeff[0]*coeff[2]
    if trip == 0.:
      return np.full(3, -coeff[1] / 3.*coeff[0])
    else:
      doub = (9.*coeff[0]*coeff[3] - coeff[1]*coeff[2]) / (2.*trip)
      sing = (4.*np.prod(coeff[:-1]) - 9.*coeff[3]*coeff[0]**2 - coeff[1]**3) / (coeff[0]*trip)
      return np.array([doub, doub, sing])
  else: # three real roots if descrim > 0 or one real, two complex conjugate roots if descrim < 0
    # i wanted to write this myself but do NOT want to work with imaginary numbers, sorry
    roots = np.roots(coeff)
    return roots[np.isreal(roots)].real

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
# endregion

# region Chemistry
def antoine_T(v: npt.NDArray, P: npt.NDArray) -> npt.NDArray:
  '''
  Calculates the temperature of every component for each pressure.
  '''
  v = np.atleast_1d(v); P = np.atleast_1d(P)
  return (-v[:, 1] / (np.log10(P) - np.r_[v[:, 0]])) - v[:, 2]

def antoine_P(v: npt.NDArray, T: npt.NDArray) -> npt.NDArray:
  '''
  Calculates the pressure of every component for each temperature.
  '''
  v = np.atleast_1d(v); T = np.atleast_1d(T)
  return 10 ** (np.c_[v[:, 0]] - np.c_[v[:, 1]] / (T + np.c_[v[:, 2]]))

def raoult_XtoY(x: list, K: list) -> tuple[npt.NDArray, float]:
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

def raoult_YtoX(y: list, K: list) -> tuple[npt.NDArray, float]:
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
# endregion

# region Iterative Tools
def lin_estimate_error(x_pair: npt.NDArray, y_pair: npt.NDArray, tol: float = 1e-10) -> npt.NDArray:
  '''
  Calculates the x-intercept (y == 0) for pairs of x and y distances. Assumes linearity.
  Expects 2D arrays, of shape N x 2.
  '''
  x_pair = np.atleast_2d(x_pair).reshape(-1, 2); y_pair = np.atleast_2d(y_pair).reshape(-1, 2)
  converged = np.all((y_pair[:, 0] == y_pair[:, 1], y_pair[:, 0] < tol), axis=0)
  skipeval = x_pair[converged, 0]; y_pair[converged, :] = np.NaN
  x_new = x_pair[:, 0] - y_pair[:, 0] * ((x_pair[:, 1] - x_pair[:, 0])/(y_pair[:, 1] - y_pair[:, 0]))
  if np.any(np.isnan(x_new)) and len(skipeval) == 0:
    raise ValueError(f"Cannot minimize error between a point and itself: x == {list(x_pair[np.isnan(x_new), 0])}")
  x_new[np.isnan(x_new)] = skipeval; y_pair[np.isnan(y_pair)] = 0.
  return x_new

def err_reduc(err_calc: Callable[[npt.NDArray], npt.NDArray], x: npt.NDArray, tol: float = 1e-10) -> tuple[npt.NDArray, npt.NDArray]:
  '''
  Evaluates an error calculation for pairs of inputs and returns new sets of inputs with a smaller average error.
  Expects 2D arrays, of shape N x 2.
  '''
  x = np.atleast_2d(x).reshape(-1, 2)
  err = np.atleast_2d(err_calc(x)) # forcing to 2D may not be necessary, but may help combat forgetfullness
  xnew = lin_estimate_error(x, err, tol)
  err = np.abs(err)
  x[(np.arange(x.shape[0]), err.argmax(axis=1))] = xnew
  return x, err

def err_reduc_iterative(err_calc: Callable[[float], float], x: npt.NDArray, tol: float = 1e-10) -> tuple[npt.NDArray, npt.NDArray, int]:
  '''
  Accepts pairs of inputs and an error function. Returns inputs with tolerable error, the errors, and the number of iterations required.
  Expects 2D arrays, of shape N x 2.
  '''
  x = np.atleast_2d(x).reshape(-1, 2)
  error = np.full_like(x, 10000.)
  i = 0
  while np.any(np.min(error, axis=1) > tol):
    x, error = err_reduc(err_calc, x, tol)
    i += 1
  return x[(np.arange(x.shape[0]), error.argmin(axis=1))], np.min(error, axis=1), i
# endregion
