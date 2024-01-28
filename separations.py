import numpy as np
import pandas as pd
import common
import numpy.typing as npt
# numb_comp, x = common.input_mol_frac()
# 
# ant_coeff = common.input_antoine(numb_comp)
# 
# P = common.input_P()

# umb_comp, x, ant_coeff, P = common.testvars()
def iterate_bubble(x: npt.ArrayLike, K: npt.ArrayLike) -> (npt.ArrayLike, float):
   '''Bubble point calculation intended to be used with DePriester Charts
   
   Parameters
   ---------
   x: ArrayLike
   Liquid mole fractions of the mixed phase feed
   K: ArrayLike 
   K values associated with the proposed temperature
   Returns
   ---------
   y: ArrayLike
   vapor mole fractions of the mixed phase feed
   error: float
   the associated error of proposed temperature
   '''
   y = np.c_[x] * K
   error = np.sum(y) - 1
   return y, error

def antoine_bubble_point(x, ant_coeff, P, tol=.05):
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

  bubbleP = T[np.argmin(error)]
  Pvap, k, y, error = calcs(bubbleP)
  return bubbleP, Pvap[:, 0], k[:, 0], y[:, 0], error[0], i

# bubbleP, Pvap, k, y, error, iters = bubble_point_iter(x, ant_coeff, P)

# print('---')
# print(f'The Bubble Temperature is {bubbleP} Celcius')
# print(f'The Error of the Calculation is {error}')
# print(f'The Calculation took {iters} iterations')
# print('---')z
# df = pd.DataFrame({'Liq Mol Fraction' : x ,
#                    'Vap Mol Fraction' : y,
#                    'Pvap (mmHg)' : Pvap,
#                    'K' : k})
# print(df)
# TODO : If we want the function to return the as is right now, or a formatted dataframe... I dont see *too* much benefit to the dataframe approach

def psi_solver(x: npt.ArrayLike, K: npt.ArrayLike, psi: float, tol: float = 0.01) -> (float, int, npt.ArrayLike) :
    '''
    Solves for vapor/liquid feed ratio (psi).
    
    Parameters
    ----------
    x : ArrayLike
        An array of the incoming feed mole fractions. Must match the size of K.
    K : ArrayLike
        An array of the respective equilibrium constants, K.
    psi : float
        An intial guess of what the ratio will be.

    Returns
    ----------
    psi : float
        The final converged value of the vapor/liquid feed ratio.
    err : float
        The associated error of the final iteration.
    x_out : ArrayLike
        the mol fractions of the outgoing liquid stream. 
    y_out : ArrayLike
        the mol fractions of the outgoing vapor stream.
    '''
    def f(psi):
        return np.sum( (x * (1- K)) / (1 + psi * (K - 1)) )
    def f_prime(psi):
        return np.sum( (x * (1 - K)**2) / (1 + psi * (K - 1))**2)
    def psi_next(psi):
        return psi - (f(psi) / f_prime(psi))
    def error(psi):
        return (psi_next(psi) - psi) / psi
    
    table = np.array([['psi', 'f', 'f_prime', 'psi_next', 'error']], dtype='object')
    
    i = 0
    while tol < np.abs(f(psi)) :
        psi = psi_next(psi)
        i += 1
    err = error(psi)
    x_out = x / (1 + psi * (K - 1))
    y_out = (x * K) / (1 + psi * (K - 1))
    return psi, i, err, x_out, y_out
    
# numb_comp, x = common.input_mol_frac()

# k = common.input_Kfac(numb_comp)

# psi_i = float(input('Input an initial guess for psi : '))

# psi, iterations, table = psi_solver(x, k, psi_i)
# print(' ---- ')
# print(f'The converged value of psi is : {psi}')
# print(f'This was done in {iterations} iterations .')
# print('----')
# print(pd.DataFrame(table))

# input('Exit? Y/N ')    

# numb_comp, x = common.input_mol_frac()
# 
# ant_coeff = common.input_antoine(numb_comp)
# 
# P = common.input_P()

# numb_comp, y, ant_coeff, P = common.testvars()
def iterate_dew(y: npt.ArrayLike, K: npt.ArrayLike) -> (npt.ArrayLike, float):
   '''Dew point calculation intended to be used with DePriester Charts
   
   Parameters
   ---------
   y: ArrayLike
   Vapor mole fractions of the mixed phase feed.
   K: ArrayLike 
   K values associated with the proposed temperature.
   Returns
   ---------
   x: ArrayLike
   Liquid mole fractions of the mixed phase feed.
   error: float
   the associated error of proposed temperature.
   '''
   x = np.c_[y] / K
   error = np.sum(x) - 1
   return x, error

def antoine_dew_point(y: npt.ArrayLike, ant_coeff: npt.ArrayLike, P: float, tol: float=.05):
  '''
  Numerical dew point calculation based on the antoine equation.
  
  Parameters
  ----------
  y : ArrayLike
  Vapor mole fractions of the two phase condition.
  ant_coeff : ArrayLike
  array of antoine coefficients associated with the components.
  P : float
  System pressure.
  tol : float
  The required error the final iteration must be under.
  '''
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

  dewP = T[np.argmin(error)]
  Pvap, k, x, error = calcs(dewP)
  return dewP, Pvap[:, 0], k[:, 0], x[:, 0], error[0], i

# dewP, Pvap, k, x, error, iters = dew_point_iter(y, ant_coeff, P)

# print('---')
# print(f'The Bubble Temperature is {dewP} Celcius')
# print(f'The Error of the Calculation is {error}')
# print(f'The Calculation took {iters} iterations')
# print('---')
# df = pd.DataFrame({'Liq Mol Fraction' : x ,
#                    'Vap Mol Fraction' : y,
#                    'Pvap (mmHg)' : Pvap,
#                    'K' : k})
# print(df)