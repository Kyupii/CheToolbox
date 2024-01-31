import numpy as np
from numpy import typing as npt

def bp_est(g : npt.ArrayLike) -> float:
  '''
  Estimates the boiling point of a compound via the group contribution method.

  Parameters:
  -----------
  g : ArrayLike
    The frequency of a group's appearance and the group's contribution value. Shape must be N x 2.
      Ex) For a molecule containing 4 groups: np.array([[3, 1.233], [1, 23.5], [2, 44.6], [7, 103.6]])
  
  Retruns:
  -----------
  T_b : float
    Estimated boiling point temperature in K (Kelvin).
  '''
  T_b = 198.2 + np.sum(g[:,0] * g[:,1])
  if T_b < 700:
    return T_b - 94.84 + 0.5577 * T_b - 0.0007705 * T_b**2
  else:
    return T_b + 282.7 - 0.5209 * T_b

def mp_est(T_b: float)-> float:
  '''
  Estimates the melting point of a compound from its boiling point.

  Parameters:
  -----------
  T_b : float
    Boiling point temperature in K (Kelvin).

  returns:
  -----------
  T_m : float
    Estimated melting point temperature in K (Kelvin).
  '''
  return 0.5839 * T_b

def pvap_solid_est(T: float, T_b: float)-> float:
  '''
  Estimates the vapor pressure of a solid compound.

  Parameters:
  -----------
  T : float
    Current temperature of the solid compound in K (Kelvin).
  T_b : float
    Boiling point temperature of the solid compound in K (Kelvin).

  Returns:
  -----------
  pvap : float
    Estimated vapor pressure in atm (atmospheres).
  '''
  T_m = mp_est(T_b)
  return -(4.4 + np.log(T_b))* (1.803 * (T_b / (T-1) - 0.803* np.log(T_b/T))) - 6.8 * (T_m / (T-1) )

def pvap_liq_est(T: float, T_b: float, K_F: float)-> float:
  '''
  Estimates the vapor pressure of a liquid compound.

  Parameters:
  -----------
  T : float
    Current temperature of the liquid compound in K (Kelvin).
  T_b : float
    Boiling point temperature of the liquid compound in K (Kelvin).
  K_F : float
    K factor of the liquid compound.

  Returns:
  -----------
  pvap : float
    Estimated vapor pressure in atm (atmospheres).
  '''
  R = 1.987
  C = -18 + 0.19 * T_b
  A = K_F * (8.75 + R * np.log( T_b ))
  return np.e ** ( (A * (T_b - C)**2 / (0.97 * R * T_b)) * ( 1 / (T_b - C) - 1 / (T-C)))

def Kow_est(f: npt.ArrayLike, c: npt.ArrayLike = np.array([[0],[0]])) -> float:
  '''
  Estimate the octanol / water equilibrium constant of a compound via the group contribution method.

  Parameters:
  -----------
  g : ArrayLike
    The frequency of a group's appearance, the group's contribution value, and the group's correction factor if necessary (0 for no correction). Shape must be N x 3.
      Ex) For a molecule containing 4 groups: np.array([[3, 1.233, 0], [1, 23.5, 0], [2, 44.6, 8.6], [7, 103.6, 13]])
  
  Returns:
  --------
  K_ow : float
    Estimated octanol / water equilibrium constant.
  '''
  return 0.229 + np.sum(f[:,0] * f[:,1]) + np.sum(f[:,0] * f[:,2])
