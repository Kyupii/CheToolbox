import numpy as np
import numpy.typing as npt
from chetoolbox import common

def raoult_XtoY(x: list, K: list) -> (npt.ArrayLike, float):
  '''
  Calculates the vapor mole fraction of a multi-component mixed phase feed (assuming liquid and gas ideality).

  Parameters
  ---------
  x : list
    Component mole fractions of the feed's liquid phase. Must sum to 1.
  K : list 
    Equalibrium constant for each component at a specific temperature and pressure. Length must equal x.
  
  Returns
  ---------
  y : ArrayLike
    Component mole fractions of the feed's vapor phase.
  error : float
    Error of calculated vapor phase component mole fractions.
  '''
  x = np.atleast_1d(x)
  K = np.atleast_1d(K)
  y = np.c_[x] * K
  error = np.sum(y) - 1
  return y, error

def raoult_YtoX(y: list, K: list) -> (npt.ArrayLike, float):
  '''
  Calculates the liquid mole fraction of a multi-component mixed phase feed (assuming liquid and gas ideality).

  Parameters
  ---------
  y : list
    Component mole fractions of the feed's vapor phase. Must sum to 1.
  K : list 
    Equalibrium constant for each component at a specific temperature and pressure. Length must equal y.
    
  Returns
  ---------
  x : ArrayLike
    Component mole fractions of the feed's liquid phase.
  error : float
    Error of calculated liquid phase component mole fractions.
  '''
  y = np.atleast_1d(y)
  K = np.atleast_1d(K)
  x = np.c_[y] / K
  error = np.sum(x) - 1
  return x, error

def psi_solver(x: list, K: list, psi: float, tol: float = 0.01) -> (float, npt.ArrayLike, npt.ArrayLike, float, int):
  '''
  Iteratively solves for the vapor/liquid output feed ratio psi (Ψ) of a multi-component liquid input stream entering a flash drum.  
  
  Parameters
  ----------
  x : list
    Component mole fractions of the liquid input feed. Must sum to 1.
  K : list
    Equalibrium constant for each component at specific temperature and pressure. Length must equal x.
  psi : float
    Initial value of psi to begin iterating on. Must be 0 <= psi <= 1.
  tol : float
    Largest error value to stop iterating and return.

  Returns
  ----------
  psi : float
    Converged vapor/liquid output feed ratio psi (Ψ).
  x_out : ArrayLike
    Component mole fractions of the output liquid stream. 
  y_out : ArrayLike
    Component mole fractions of the output vapor stream.
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

def bubble_point(x: list, ant_coeff: npt.ArrayLike, P: float, tol: float = .05) -> (float, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, float, int):
  '''
  Iteratively solves for the bubble point temperature of a multi-component liquid mixture.

  Parameters
  ----------
  x : list
    Component mole fractions of the liquid mixture. Must sum to 1.
  ant_coeff : ArrayLike
    Components' coefficients for the Antoine Equation of State. Shape must be N x 3.
  P : float
    Ambient pressure of the liquid mixture in mmHg (millimeters of mercury).
  tol : float
    Largest error value to stop iterating and return.

  Returns
  ----------
  bubbleT : float
    Temperature of the liquid mixture's bubble point in C (Celcius).
  Pvap : ArrayLike
    Vapor pressure for each component at the bubble point temperature. 
  K : ArrayLike
    Equalibrium constant for each component at the stated pressure and bubble point temperature. 
  y : ArrayLike
    Component mole fractions of the first bubble of vapor.
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

def dew_point(y: list, ant_coeff: npt.ArrayLike, P: float, tol: float = .05) -> (float, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, float, int):
  '''
  Iteratively solves for the dew point temperature of a multi-component vapor mixture.

  Parameters
  ----------
  y : list
    Component mole fractions of the vapor mixture. Must sum to 1.
  ant_coeff : ArrayLike
    Components' coefficients for the Antoine Equation of State. Shape must be N x 3.
  P : float
    Ambient pressure of the vapor mixture in mmHg (millimeters of mercury).
  tol : float
    Largest error value to stop iterating and return.

  Returns
  ----------
  dewT : float
    Temperature of the vapor mixture's dew point in C (Celcius).
  Pvap : ArrayLike
    Vapor pressure for each component at the dew point temperature. 
  K : ArrayLike
    Equalibrium constant for each component at the stated pressure and dew point temperature. 
  x : ArrayLike
    Component mole fractions of the first dew of liquid.
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
