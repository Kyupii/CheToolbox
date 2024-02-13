import numpy as np
import numpy.typing as npt
from chetoolbox import common
from common import LinearEq

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

def psi_solver(x: list, K: list, psi: float, tol: float = 0.01) -> tuple[float, npt.ArrayLike, npt.ArrayLike, float, int]:
  '''
  Iteratively solves for the vapor/liquid output feed ratio psi (Ψ) of a multi-component liquid input stream entering a flash drum.  
  
  Parameters
  ----------
  x : list
    Component mole fractions of the liquid input feed (unitless). Must sum to 1.
  K : list
    Equalibrium constant for each component at specific temperature and pressure (units vary). Length must equal x.
  psi : float
    Initial value of psi to begin iterating on (unitless). Must be 0 <= psi <= 1.
  tol : float
    Largest error value to stop iterating and return.

  Returns
  ----------
  psi : float
    Converged vapor/liquid output feed ratio psi (Ψ) (unitless).
  x_out : ArrayLike
    Component mole fractions of the output liquid stream (unitless). 
  y_out : ArrayLike
    Component mole fractions of the output vapor stream (unitless).
  error : float
    Error at the final iteration.
  i : int
    Number of iterations to calculate to the specified tolerance.
  '''
  x = np.atleast_1d(x)
  K = np.atleast_1d(K)
  def f(psi):
    return np.sum( (x * (1 - K)) / (1 + psi * (K - 1)) )
  def f_prime(psi):
    return np.sum( (x * (1 - K)**2) / (1 + psi * (K - 1))**2)
  def psi_next(psi):
    return psi - (f(psi) / f_prime(psi))
  def error(psi):
    return np.abs(f(psi))
  
  i = 0
  while error(psi) > tol:
    psi = psi_next(psi)
    i += 1
  x_out = x / (1 + psi * (K - 1))
  y_out = (x * K) / (1 + psi * (K - 1))
  return psi, x_out, y_out, error(psi), i

def bubble_point(x: list, ant_coeff: npt.ArrayLike, P: float, tol: float = .05) -> tuple[float, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, float, int]:
  '''
  Iteratively solves for the bubble point temperature of a multi-component liquid mixture.

  Parameters
  ----------
  x : list
    Component mole fractions of the liquid mixture (unitless). Must sum to 1.
  ant_coeff : ArrayLike
    Components' coefficients for the Antoine Equation of State (unitless). Shape must be N x 3.
  P : float
    Ambient pressure of the liquid mixture in mmHg (millimeters of mercury).
  tol : float
    Largest error value to stop iterating and return.

  Returns
  ----------
  bubbleT : float
    Temperature of the liquid mixture's bubble point in C (Celcius).
  Pvap : ArrayLike
    Vapor pressure for each component at the bubble point temperature in mmHg (millimeters of mercury). 
  K : ArrayLike
    Equalibrium constant for each component at the stated pressure and bubble point temperature (units vary). 
  y : ArrayLike
    Component mole fractions of the first bubble of vapor (unitless).
  error : float
    Error at the final iteration.
  i : int
    Number of iterations to calculate to the specified tolerance.
  '''
  x = np.atleast_1d(x)
  ant_coeff = np.atleast_1d(ant_coeff).reshape(-1, 3)
  boil_points = common.antoine_T(ant_coeff, P)
  T = [np.max(boil_points), np.min(boil_points)]

  def calcs(T):
    Pvap = common.antoine_P(ant_coeff, T)
    k = Pvap / P
    y = np.c_[x] * k
    error =  np.sum(y, axis=0) - 1
    return Pvap, k, y, error

  def iter(T):
    _, _, _, error = calcs(T)
    Tnew = common.lin_estimate_error(T, error)
    error = np.abs(error)
    T[np.argmin(error)] = Tnew
    return error, T
  
  error = 10000
  i = 0
  while np.min(error) > tol:
    error, T = iter(T)
    i += 1

  bubbleT = T[np.argmin(error)]
  Pvap, k, y, error = calcs(bubbleT)
  return bubbleT, Pvap[:, 0], k[:, 0], y[:, 0], error[0], i

def dew_point(y: list, ant_coeff: npt.ArrayLike, P: float, tol: float = .05) -> tuple[float, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, float, int]:
  '''
  Iteratively solves for the dew point temperature of a multi-component vapor mixture.

  Parameters
  ----------
  y : list
    Component mole fractions of the vapor mixture (unitless). Must sum to 1.
  ant_coeff : ArrayLike
    Components' coefficients for the Antoine Equation of State (unitless). Shape must be N x 3.
  P : float
    Ambient pressure of the vapor mixture in mmHg (millimeters of mercury).
  tol : float
    Largest error value to stop iterating and return.

  Returns
  ----------
  dewT : float
    Temperature of the vapor mixture's dew point in C (Celcius).
  Pvap : ArrayLike
    Vapor pressure for each component at the dew point temperature in mmHg (millimeters of mercury). 
  K : ArrayLike
    Equalibrium constant for each component at the stated pressure and dew point temperature (units vary). 
  x : ArrayLike
    Component mole fractions of the first dew of liquid (unitless).
  error : float
    Error at the final iteration.
  i : int
    Number of iterations to calculate to the specified tolerance.
  '''
  y = np.atleast_1d(y)
  ant_coeff = np.atleast_1d(ant_coeff).reshape(-1, 3)
  boil_points = common.antoine_T(ant_coeff, P)
  T = [np.max(boil_points), np.min(boil_points)]

  def calcs(T):
    Pvap = common.antoine_P(ant_coeff, T)
    k = Pvap / P
    x = np.c_[y] / k
    error =  np.sum(x, axis=0) - 1
    return Pvap, k, x, error
  
  def iter(T):
    _, _, _, error = calcs(T)
    Tnew = common.lin_estimate_error(T, error)
    error = np.abs(error)
    T[np.argmin(error)] = Tnew
    return error, T
  
  error = 10000
  i = 0
  while np.min(error) > tol:
    error, T = iter(T)
    i += 1

  dewT = T[np.argmin(error)]
  Pvap, k, x, error = calcs(dewT)
  return dewT, Pvap[:, 0], k[:, 0], x[:, 0], error[0], i

def liq_frac_subcooled(Cpl: float, heatvap: float, Tf: float, Tb: float) -> float:
  '''
  Calculates the liquid fraction of a subcooled bianary liquid mixture feed.

  Parameters:
  -----------
  Cpl : float
    Specific heat of the liquid feed in kJ/mol*K (kilojoules per mole Kelvin).
  heatvap : float
    Heat of vaporization of the liquid feed in kJ/mol (kilojoules per mole).
  Tf : float
    Temperature of the liquid feed in K (Kelvin).
  Tb : float
    Bubble temperature of the liquid feed in K (Kelvin).

  Returns:
  -----------
  q : float
    Liquid fraction of the feed (unitless).
  '''
  return 1. + Cpl * (Tb - Tf) / heatvap

def liq_frac_superheated(Cpv: float, heatvap: float, Tf: float, Td: float) -> float:
  '''
  Calculates the liquid fraction of a superheated bianary vapor mixture feed.

  Parameters:
  -----------
  Cpv : float
    Specific heat of the vapor feed in kJ/mol*K (kilojoules per mole Kelvin).
  heatvap : float
    Heat of vaporization of the vapor feed in kJ/mol (kilojoules per mole).
  Tf : float
    Temperature of the vapor feed in K (Kelvin).
  Td : float
    Dew temperature of the vapor feed in K (Kelvin).

  Returns:
  -----------
  q : float
    Liquid fraction of the feed (unitless).
  '''
  return 1. + Cpv * (Tf - Td) / heatvap

def tops_bottom_split(F: float, xf: float, xd: float, xb: float) -> float:
  '''
  Calculates the distilate and bottoms flow rates out of a bianary mixture distilation column.

  Parameters:
  -----------
  F : float
    Feed molar flowrate in kmol/hr (kilomoles per hour).
  xf : float
    Liquid fraction of the lower boiling boint species in the feed (unitless).
  xd : float
    Liquid fraction of the lower boiling boint species in the distilate (unitless).
  xb : float
    Liquid fraction of the lower boiling boint species in the bottoms (unitless).

  Returns:
  -----------
  D : float
    Distilate molar flowrate in kmol/hr (kilomoles per hour).
  B : float
    Bottoms molar flowrate in kmol/hr (kilomoles per hour).
  '''
  D = F*(xf - xd)/(xd - xb)
  return D, F - D

def feedline_graph(q: float, xf: float) -> LinearEq:
  '''
  Calculates the slope and intercepts of the feed line on a McCabe Thiel Diagram for a bianary mixture distilation column.

  Parameters:
  -----------
  q : float
    Feed liquid fraction (unitless).

  Returns:
  -----------
  m : float
    Slope of the feed line (unitless).
  y_int : float
    Y-intercept of the feed line (unitless).
  x_int : float
    X-intercept of the feed line (unitless).
  '''
  if q == 1: # vertical feed line
    m = np.NaN
    y_int = np.NaN
    x_int = xf
  else:
    m = -q / (1. - q)
    y_int = m*xf + xf
    x_int = -y_int / m
  return LinearEq(m, y_int, x_int)

# TODO under construction
def mccabe_thiel_graph(feedline: LinearEq, feedpoint_eq: tuple, xf: float, xd: float, xb: float, Rmin_mult: float) -> tuple[float, float, LinearEq]:
  # TODO complete this
  '''
  Calculates a McCabe Thiel Diagram for a bianary mixture distilation column.

  Parameters:
  -----------
  feedEQ : tuple
    Point of intersection between the feed line and the equalibrium line on a McCabe Thiel Diagram (unitless, unitless). Bounded [0, 1]. Length must equal 2.
  xf : float
    Liquid fraction of the lower boiling boint species in the feed (unitless).
  xd : float
    Liquid fraction of the lower boiling boint species in the distilate (unitless).
  xb : float
    Liquid fraction of the lower boiling boint species in the bottoms (unitless).

  Returns:
  -----------

  '''
  xf_eq, yf_eq = feedpoint_eq
  # "distilate to feed at equalibrium" line
  m = (xd - yf_eq) / (xd - xf_eq) 

  # "distilate to feedpoint" line
  Rmin = m / (1. - m)
  R = Rmin_mult * Rmin
  m = R / (1. + R)
  y_int = xd / (1. + R)
  x_int = -y_int / m
  distline = LinearEq(m, y_int, x_int)

  # feedpoint
  feedpoint = common.linear_intersect(feedline, distline)

  # bottoms to feed point
  m = (x - xb) / (y - xb) 


  return

  

# TODO under construction
def mccabe_thiel(F: float, xf: float, xd: float, xb: float, q: float, yf_eq: float, Rmin_mult: float,):
  '''
  Calculates various design parameters of a bianary mixture distilation column using McCabe Thiel Diagram analysis. Assumes equal molar heats of vaporization.

  Parameters:
  -----------
  F : float
    Feed molar flowrate in kmol/hr (kilomoles per hour).
  xf : float
    Concentration of the lower boiling boint species in the feed (unitless).
  T : float
    Current temperature of the reaction in K (Kelvin).

  Returns:
  -----------
  K : float
    Equilibrium Constant (units vary).
  '''
  D, B = tops_bottom_split(F, xf, xd, xb)
  if q == 1: Sf = np.NaN
  else: Sf = -q / (1. - q)


  return