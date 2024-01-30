# This is the icp class of the module
import numpy as np
from numpy import typing as npt
def est_bp(g : npt.ArrayLike) -> float:
    '''Estimate the normal boiling point by group contribution method (Deg K)
    Parameters:
    -----------
    g : ArrayLike
    an N x 2 array of the number of groups type i, and their respective group contribution
    Retruns:
    -----------
    T_b : Normal boiling temperature in kelvin
    '''
    T_b = 198.2 + np.sum(g[:,0] * g[:,1])
    if T_b < 700:
        T_b = T_b - 94.84 + 0.5577 * T_b - 0.0007705 * T_b**2
    else:
        T_b + 282.7 - 0.5209 * T_b
def est_mp(T_b: float)-> float:
    '''Estimates the melting point of a compound
    Parameters:
    -----------
    T_b : float
    The nornal boiling temperature of the compound in deg K
    returns:
    -----------
    est_mp : float
    The estimated melting temperature of the compound in deg K'''
    return 0.5839 * T_b
def pvap_solid(T: float, T_b: float)-> float:
    ''''Estimate the vapor pressure of a solid species
    Parameters:
    -----------
    T : float
    The temperature at which the vapor pressure is to be calculated in deg K
    T_b : float
    The normal boiling temperature of the compound in deg K
    returns:
    -----------
    pvap_solid : float
    The vapor pressure of the solid compound in atm
    '''
    T_m = est_mp(T_b)
    return -(4.4 + np.log(T_b))* (1.803 * (T_b / (T-1) - 0.803* np.log(T_b/T))) - 6.8 * (T_m / (T-1) )
def pvap_liq(T: float, T_b: float, K_F: float)-> float:
    '''Estimate the vapor pressure of a liquid
    Paramters
    ---------
    T : float
    Temperature at which vapor pressure is to be calculated
    T_b : float
    normal boiling point in deg K
    K_F : float
    Factor K_F from table
    Returns:
    ---------
    The estimated vapor pressure of the compound in atm'''
    R = 1.987
    C = -18 + 0.19 * T_b
    A = K_F * (8.75 + R * np.log( T_b ))
    return np.e ** ( (A * (T_b - C)**2 / (0.97 * R * T_b)) * ( 1 / (T_b - C) - 1 / (T-C)))
def est_kow(f: npt.ArrayLike, c: npt.ArrayLike = np.array([[0],[0]])) -> float:
    '''Estimate the octanol / water equilibrium constant
    Parameters:
    -----------
    f : ArrayLike
    an N x 2 array of the number of groups and their group contribution
    c : ArrayLike
    an N x 2 array of the number of groups requiring correction and their correction factor. The default is 0 for no corrections
    
    Returns:
    --------
    K_ow : float
    The octanol/water equilibrium constant'''
    # I think setting the default of c to a 1 x 1 array of 0 works when there is no correction...
    return 0.229 + np.sum(f[:,0] * f[:,1]) + np.sum(c[:,0] * c[:,1])
