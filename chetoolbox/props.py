import numpy as np
from numpy import typing as npt
import pandas as pd

def antoine_coeff_query(query: str | npt.NDArray):
  '''
  Obtains Antoine coefficients for components based on a query. 
  
  Parameters:
  -----------
  Query: Float | NDArray
    A single string query or an array of string values.
  
  Returns:
  -----------
  coeff : NDArray
    Antoine coefficients of each queried compound. Shape is N x 3.
  
  Example Usage:
  -----------
  ```py
  props.antoine_coeff(['c1', 'c2', 'ic4'])
  ```
  '''
  query = np.atleast_1d(query)
  headers = ["ID", "Formula", "Compound Name", "A", "B", "C", "TMIN", "TMAX"]
  if type(query[0]) == np.str_: #assume name
    col = headers[2]
  else: # assume ID
    col = headers[0]
  
  antoine = pd.read_csv('datasets/antoine.csv')
  coeff = np.zeros((query.shape[-1], 3))
  for i, item in enumerate(query):
    coeff[i] = antoine[antoine.loc[:, col] == item].iloc[:, 3:6].to_numpy()
  return coeff

def k_coeff_query(query: str | npt.NDArray):
  '''
  Obtains McWilliams / Almehaideb K coefficients for components based on a query. 
  
  Parameters:
  -----------
  Query: Float | NDArray
    A single string query or an array of string values.
  
  Returns:
  -----------
  coeff : NDArray
    McWilliams / Almehaideb K coefficients of each queried compound. Shape is N x 7.
  
  Example Usage:
  -----------
  ```py
  props.antoine_coeff(['c1', 'c2', 'ic4'])
  ```
  '''
  query = np.atleast_1d(query)
  headers = ["Compound Name", "aT1", "aT2", "aT3", "aP1", "aP2", "aP3", "aw"]
  col = headers[0]
  antoine = pd.read_csv('datasets/k_almehaideb.csv')
  coeff = np.zeros((query.shape[-1], 7))
  for i, item in enumerate(query):
    coeff[i] = antoine[antoine.loc[:, col] == item].iloc[:, 1:].to_numpy()
  return coeff

def convergence_P(T_and_P: npt.NDArray, MWC7p: float, sgC7p: float):
  # T must be in Rankin # P must be in psia
  T_and_P = np.atleast_1d(T_and_P).reshape(-1, 2); T = np.c_[T_and_P[:, 0]]; P = np.c_[T_and_P[:, 1]]
  linterm = -2381.8542 + 46.341487 * MWC7p * sgC7p
  ais = np.array([6124.3049, -2753.2538, 415.42049])
  sumterm = np.sum(ais*(MWC7p*sgC7p/(T - 460.))**np.arange(3), axis=1, keepdims=True)
  Pk = linterm + sumterm
  A = 1. - ((P - 14.7) / (Pk - 14.7))**.6
  return Pk, A 

def k_wilson(Pci: npt.NDArray, Tci: npt.NDArray, omega: npt.NDArray, T_and_P: npt.NDArray):
  # T must be in Rankin # P must be in psia
  Pci = np.atleast_1d(Pci); Tci = np.atleast_1d(Tci); omega = np.atleast_1d(omega)
  T_and_P = np.atleast_1d(T_and_P).reshape(-1, 2); T = np.c_[T_and_P[:, 0]]; P = np.c_[T_and_P[:, 1]]
  return (Pci / P) * np.exp(5.37 * (1. + omega) * (1. - Tci / T))

def k_whitson(Pci: npt.NDArray, Tci: npt.NDArray, omega: npt.NDArray, T_and_P: npt.NDArray, MWC7p: float, sgC7p: float):
  # T must be in Rankin # P must be in psia
  Pci = np.atleast_1d(Pci); Tci = np.atleast_1d(Tci); omega = np.atleast_1d(omega)
  T_and_P = np.atleast_1d(T_and_P).reshape(-1, 2); P = np.c_[T_and_P[:, 1]]
  PciP = Pci / P
  K = k_wilson(Pci, Tci, omega, T_and_P)
  Pk, A = convergence_P(T_and_P, MWC7p, sgC7p)
  return K**A * PciP**(1. - A) * (Pci / Pk)**(A - 1.)

def k_mcwilliams(coeffs: npt.NDArray, T_and_P: npt.NDArray):
  # T must be in Rankin # P must be in psia
  T_and_P = np.atleast_1d(T_and_P).reshape(-1, 2); T = np.c_[T_and_P[:, 0]]; P = np.c_[T_and_P[:, 1]]
  return np.exp(coeffs[:, 0] / T**2 + coeffs[:, 1] / T + coeffs[:, 2] + coeffs[:, 3] * np.log(P) + coeffs[:, 4] / P**2 + coeffs[:, 5] / P)

def k_almehaideb(coeffs: npt.NDArray, Pci: npt.NDArray, T_and_P: npt.NDArray, omega: float, MWC7p: float, sgC7p: float):
  # T must be in Rankin # P must be in psia
  coeffs = np.atleast_2d(coeffs)
  T_and_P = np.atleast_1d(T_and_P).reshape(-1, 2); P = np.c_[T_and_P[:, 1]]
  Kstar = k_mcwilliams(coeffs, T_and_P)
  
  if coeffs[-1, 6] != 0. and omega is not None: # assumes the last component is the C7+
    Kstar[:, -1] = Kstar[:, -1] * np.exp(coeffs[-1, 6] / omega)
  
  Pk, A = convergence_P(T_and_P, MWC7p, sgC7p)
  return (Pci / Pk)**(A - 1.) * (Pci / P) * Kstar**A

def bp_est(g : npt.NDArray) -> float:
  '''
  Estimates the boiling point of a compound via the group contribution method.

  Parameters:
  -----------
  g : NDArray
    The frequency of a group's appearance and the group's contribution value. Shape must be N x 2.
      Ex) For a molecule containing 4 groups: np.array([[3, 1.233], [1, 23.5], [2, 44.6], [7, 103.6]])
  
  Returns:
  -----------
  T_b : float
    Estimated boiling point temperature in K (Kelvin).
  '''
  g = np.atleast_1d(g).reshape(-1, 2)
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

def pvap_solid_est(T_b: float, T_m: float, T: float)-> float:
  '''
  Estimates the vapor pressure of a solid compound.

  Parameters:
  -----------
  T_b : float
    Boiling point temperature of the solid compound in K (Kelvin).
  T_m : float
    Estimated melting point temperature in K (Kelvin).
  T : float
    Current temperature of the solid compound in K (Kelvin).

  Returns:
  -----------
  pvap : float
    Estimated vapor pressure in atm (atmospheres).
  '''
  return np.exp( -(4.4 + np.log(T_b)) * (1.803 * (T_b / T - 1.) - 0.803 * np.log(T_b/T)) - 6.8 * (T_m / T - 1.) )

def pvap_liq_est(T_b: float, K_F: float, T: float)-> float:
  '''
  Estimates the vapor pressure of a liquid compound.

  Parameters:
  -----------
  T_b : float
    Boiling point temperature of the liquid compound in K (Kelvin).
  K_F : float
    K factor of the liquid compound.
  T : float
    Current temperature of the liquid compound in K (Kelvin).

  Returns:
  -----------
  pvap : float
    Estimated vapor pressure in atm (atmospheres).
  '''
  R = 1.987 #cal/mol*K
  C = -18. + 0.19 * T_b
  A = K_F * (8.75 + R * np.log( T_b ))
  return np.exp( (A * (T_b - C)**2 / (0.97 * R * T_b)) * ( 1. / (T_b - C) - 1. / (T - C)))

def Kow_est(f: npt.NDArray) -> float:
  '''
  Estimates the octanol / water equilibrium constant of a compound via the group contribution method.
  
  Parameters:
  -----------
  g : NDArray
    The frequency of a group's appearance, the group's contribution value, and the group's correction factor if necessary (0 for no correction). Shape must be N x 3.
      Ex) For a molecule containing 4 groups: np.array([[3, 1.233, 0], [1, 23.5, 0], [2, 44.6, 8.6], [7, 103.6, 13]])
  
  Returns:
  --------
  K_ow : float
    Estimated octanol / water equilibrium constant (unitless).
  '''
  return 10**(0.229 + np.sum(f[:,0] * f[:,1]) + np.sum(f[:,0] * f[:,2]))

def bioconc_est(K_ow: float, c: list = None) -> tuple[float, str]:
  '''
  Estimates the tissue / water bioconcentration factor of a compound.
  
  Parameters:
  -----------
  K_ow : float
    Octanol / water equilibrium constant.
  c : list
    Correction factors for specific structral groups present in the compound. 
  
  Returns:
  --------
  bcf : float
    Estimated bioconcentration factor in L/kg (liters per kilogram).
  cat : float
    Qualitative potential for tissue accumulation.
  '''
  if c is None:
    bcf = 10.**(.79 * np.log10(K_ow) - .4)
  else:
    bcf = 10.**(.77 * np.log10(K_ow) + np.sum(c) - .7)
  if bcf <= 250:    cat = "Low Potential for Tissue Accumulation"
  elif bcf <= 1000: cat = "Moderate Potential for Tissue Accumulation"
  else:             cat = "High Potential for Tissue Accumulation"
  return bcf, cat

def water_sol_est(K_ow: float, c: list = None, T_m: float = None, MW: float = None) -> tuple[float, str]: 
  '''
  Estimates the water solubility of a compound. Either T_m, MW, or both are required.
  
  Parameters:
  -----------
  K_ow : float
    Octanol / water equilibrium constant.
  c : list
    Correction factors for specific structral groups present in the compound. 
  T_m : float
    Melting point temperature in K (Kelvin).
  MW : float
    Molecular weight of the compound in kg/kmol (kilograms per kilomole). 
  
  Returns:
  --------
  sol : float
    Estimated water solubility in mol/L (moles per liter).
  cat : float
    Qualitative solubility in water.
  '''
  if T_m is None:
    sol = 10.**(.796  - .854 * np.log10(K_ow) - .00728 * MW + np.sum(c))
  elif MW is None:
    sol = 10.**(.342 - 1.0374 * np.log10(K_ow) - .0108 * (T_m - 298.15) + np.sum(c))
  else:
    sol = 10.**(.693 - .96 * np.log10(K_ow) - .0092 * (T_m - 298.15) - .00314 * MW + np.sum(c))
  ppm = sol * 35500
  if ppm <= .1 :      cat = "Insoluable in Water"
  elif ppm <= 100:    cat = "Slightly Soluable in Water"
  elif ppm <= 1000:   cat = "Moderately Soluable in Water"
  elif ppm <= 10000:  cat = "Soluable in Water"
  else:               cat = "Very Soluable in Water"
  return sol, cat

def henry_est(g : npt.NDArray, T: float = None) -> tuple[float, str | None]: 
  '''
  Estimates the Henry's Law constant of a compound by group contribution method.
  
  Parameters:
  -----------
  g : NDArray
    The frequency of a group's appearance, the group's contribution value, and the group's correction factor if necessary (0 for no correction). Shape must be N x 3.
      Ex) For a molecule containing 4 groups: np.array([[3, 1.233, 0], [1, 23.5, 0], [2, 44.6, 8.6], [7, 103.6, 13]])
  T : float
    Current temperature of the compound in K (Kelvin).
  
  Returns:
  -----------
  H : float
    Estimated Henry's Law constant of the compound (unitless).
  '''
  g = np.atleast_1d(g).reshape(-1, 3)
  H = 10 ** -(np.sum(g[:,0] * g[:,1]) + np.sum(g[:,0] * g[:,2])) #this is unitless H
  if T is None: return H, None

  R = 8.20575e-5 # atm*m^3/mol*K
  H_units = H*R*T
  if H_units <= 1.e-7 :   cat = "Non-volatile"
  elif H_units <= 1.e-5:  cat = "Slightly Volatile"
  elif H_units <= 1.e-3:  cat = "Moderately Volatile"
  elif H_units <= 1.e-1:  cat = "Volatile"
  else:                   cat = "Very Volatile"
  return H, cat

def soil_sorb_est(g: npt.NDArray, mole_con: npt.NDArray | float) -> float:
  '''
  Estimates the soil sobrtion coefficient of a compound via the group contribution method.
  
  Parameters:
  -----------
  g : NDArray
    The frequency of a group's appearance and the group's correction factor. Shape must be N x 2.
      Ex) For a molecule containing 4 groups: np.array([[3, 1.233], [1, 23.5], [2, 44.6], [7, 103.6]])
  mole_con: npt.NDArray | float
    First order molecular connectivity index of the compound. Must be precomputed (float) or of shape 1 x 2N.
      Ex) np.array([1, 3, 1, 3, 2, 3, 2, 1]) or 2.68
  
  Retruns:
  -----------
  K_oc : float
    Estimated soil sobrtion coefficient in ug*mL/g*mg (the mass ratio of compound to organic carbon in soil [mg/g] divided by the concentration of the compound in water [mg/mL]).
  '''
  if type(mole_con) == npt.NDArray:
    mole_con = np.atleast_1d(mole_con).reshape(-1, 2)
    mole_con = np.sum( (1. / (mole_con[:,0]*mole_con[:,1]) )**.5 )
  
  g = np.atleast_1d(g).reshape(-1, 2)
  return 10**(.53 * mole_con + .62 + np.sum( g[:,0] * g[:,1] ))

def biodegrade_est(g: npt.NDArray, MW: float) -> tuple[float, str]:
  '''
  Estimates the biodegradation index of a compound via the group contribution method.
  
  Parameters:
  -----------
  g : NDArray
    The frequency of a group's appearance and the group's contribution value. Shape must be N x 2.
      Ex) For a molecule containing 4 groups: np.array([[3, 1.233], [1, 23.5], [2, 44.6], [7, 103.6]])
  MW: npt.NDArray | float
    Molecular weight of the compound in kg/kmol (kilograms per kilomole).
  
  Retruns:
  -----------
  I : float
    Estimated biodegradation index (unitless).
  cat : float
    Qualitative rate of aerobic biodegradation, as an estimated lifetime.
  '''
  g = np.atleast_1d(g).reshape(-1, 2)
  I = 3.199 + np.sum( g[:,0] * g[:,1] ) - .00221*MW
  Iround = np.round(I, 0)
  if Iround <= 1. :   cat = "Lifetime of Years"
  elif Iround == 2.:  cat = "Lifetime of Months"
  elif Iround == 3.:  cat = "Lifetime of Weeks"
  elif Iround == 4.:  cat ="Lifetime of Days"
  else: cat = "Lifetime of Hours"
  
  return I , cat

def cp_est(const: list, T: float) -> float:
  '''
  Estimates the specific heat capacity of a compound.
  
  Parameters:
  -----------
  const : list
    The compound's A, B, C, D specific heat constants. A is the vapor phase average specific heat in J/mol*K (Joule per mole Kelvin). Length must be 4.
  T : float
    Current temperature of the compound in K (Kelvin).

  Returns:
  -----------
  Cp : float
    Estimated specific heat capacity of the compound in J/mol*K (Joules per mole Kelvin).
  '''
  return const[0] + const[1]*T + const[2]*T**2 + const[3] / T**2

# TODO #13 function to solve for ABCD specific heat constants given multiple (T, Cp) pairs

def hess(prod: npt.NDArray, reac: npt.NDArray):
  '''
  Estimates change in enthalpy, entropy, or Gibb's free energy based on formation values for delta H, delta S or delta G.
  
  Parameters:
  -----------
  prod : NDArray
    The product species' stoichiometric coefficient and delta H, S, or G for formation in J/mol (Joules per mole). Shape must be N x 2.
      Ex) np.array([coeff1, A1], [coeff2, A2], [coeff3, A3])
  reac : NDArray
    The reactant species' stoichiometric coefficient and delta H, S, or G for formation in J/mol (Joules per mole). Shape must be N x 2.
      Ex) np.array([coeff1, A1], [coeff2, A2], [coeff3, A3])
  Returns:
  -----------
  delta : float
    change in enthalpy, entropy, or Gibb's free energy for the reaction in J/mol (Joules per mole).
  '''
  prod = np.atleast_1d(prod).reshape(-1, 2)
  reac = np.atleast_1d(reac).reshape(-1, 2)

  return np.sum(prod[:,0]*prod[:,1]) - np.sum(reac[:,0]* reac[:,1])

def deltaH_est(prod: npt.NDArray, reac: npt.NDArray, T: float, deltaH_0: float, T_0: float = 298.) -> float:
  '''
  Estimates enthalpy of a balanced chemical reaction. A negative value indicates a net energy release.
  
  Parameters:
  -----------
  prod : NDArray
    The product species' stoichiometric coefficient and A, B, C, D specific heat constants. A is the vapor phase average specific heat in J/mol*K (Joule per mole Kelvin). Shape must be N x 5.
      Ex) np.array([coeff1, A1, B1, C1, D1], [coeff2, A2, B2, C2, D2], [coeff3, A3, B3, C3, D3])
  reac : NDArray
    The reactant species' stoichiometric coefficient and A, B, C, D specific heat constants. A is the vapor phase average specific heat in J/mol*K (Joules per mole Kelvin). Shape must be N x 5.
      Ex) np.array([coeff1, A1, B1, C1, D1], [coeff2, A2, B2, C2, D2], [coeff3, A3, B3, C3, D3])
  T : float
    Current temperature of the reaction in K (Kelvin).
  deltaH_0 : float
    Standard heat of reaction in J/mol (Joules per mole).
  T_0 : float
    Standard reference temperature in K (Kelvin). Most standard heat of formation tables are calcualted at 298 K (Kelvin).

  Returns:
  -----------
  deltaH : float
    Estimated enthalpy of the reaction in J/mol (Joules per mole).
  '''
  phi = T / T_0
  prod = np.atleast_1d(prod).reshape(-1, 5)
  reac = np.atleast_1d(reac).reshape(-1, 5)
  delta = np.sum(np.c_[prod[:, 0]]*prod[:, 1:], axis=0) - np.sum(np.c_[reac[:, 0]]*reac[:, 1:], axis=0)
  # deltaH = deltaH_0 + integral( Cp dT) = integral( A + B*T + C*T**2 + D/T**2 dT) = AT + .5*B*T**2 + .33*C*T**3 - D/T
  return deltaH_0 + delta[0] * (T - T_0) + delta[1] / 2. * (T**2 - T_0**2) + delta[2] / 3. * (T**3 - T_0**3) + delta[3] / T_0 * ((phi - 1.) / phi)

def deltaS_est(prod: npt.NDArray, reac: npt.NDArray, T: float, deltaS_0: float, T_0: float = 298.) -> float:
  '''
  Estimates entropy of a balanced chemical reaction. A negative value indicates a net increase in order.
  
  Parameters:
  -----------
  prod : NDArray
    The product species' stoichiometric coefficient and A, B, C, D specific heat constants. A is the vapor phase average specific heat in J/mol*K (Joule per mole Kelvin). Shape must be N x 5.
      Ex) np.array([coeff1, A1, B1, C1, D1], [coeff2, A2, B2, C2, D2], [coeff3, A3, B3, C3, D3])
  reac : NDArray
    The reactant species' stoichiometric coefficient and A, B, C, D specific heat constants. A is the vapor phase average specific heat in J/mol*K (Joule per mole Kelvin). Shape must be N x 5.
      Ex) np.array([coeff1, A1, B1, C1, D1], [coeff2, A2, B2, C2, D2], [coeff3, A3, B3, C3, D3])
  T : float
    Current temperature of the reaction in K (Kelvin).
  deltaS_0 : float
    Standard entropy of reaction in J/mol*K (Joules per mole Kelvin).
  T_0 : float
    Standard reference temperature in K (Kelvin). Most standard heat of formation tables are calcualted at 298 K (Kelvin).

  Returns:
  -----------
  deltaS : float
    Estimated entropy of reaction in J/mol*K (Joules per mole Kelvin).
  '''
  prod = np.atleast_1d(prod).reshape(-1, 5)
  reac = np.atleast_1d(reac).reshape(-1, 5)
  delta = np.sum(np.c_[prod[:, 0]]*prod[:, 1:], axis=0) - np.sum(np.c_[reac[:, 0]]*reac[:, 1:], axis=0)
  # deltaS = deltaS_0 + integral( Cp / T dT) = integral( A/T + B + C*T + D/T**3 dT) = Aln(T) + B*T + .5*C*T**2 -.5*D/T**2
  return deltaS_0 + delta[0] * np.log(T/T_0) + delta[1] * (T - T_0) + .5 * delta[2] * (T**2 - T_0**2) - .5 * delta[3] / (T**2 - T_0**2)

def gibbs_rxn(deltaH: float, deltaS: float, T: float) -> float:
  '''
  Calculates Gibbs free energy of a balanced chemical reaction from the reaction's net change in enthalpy and entropy. A negative value indicates a spontaneous reaction.
  
  Parameters:
  -----------
  deltaH : float
    Heat of reaction in J/mol (Joules per mole).
  deltaS : float
    Entropy of reaction in J/mol*K (Joules per mole Kelvin).
  T : float
    Current temperature of the reaction in K (Kelvin).

  Returns:
  -----------
  deltaG : float
    Estimated Gibbs free energy of the reaction in J/mol (Joules per mole).
  '''
  return deltaH - T * deltaS

def gibbs_est(prod: npt.NDArray, reac: npt.NDArray, T: float, deltaH_0: float, deltaG_0: float, T_0: float = 298.) -> float:
  '''
  Estimates Gibbs free energy of a balanced chemical reaction via its standard net change in enthalpy and Gibbs free energy. A negative value indicates a spontaneous reaction.
  
  Parameters:
  -----------
  prod : NDArray
    The product species' stoichiometric coefficient and A, B, C, D specific heat constants. A is the vapor phase average specific heat in J/mol*K (Joule per mole Kelvin). Shape must be N x 5.
      Ex) np.array([coeff1, A1, B1, C1, D1], [coeff2, A2, B2, C2, D2], [coeff3, A3, B3, C3, D3])
  reac : NDArray
    The reactant species' stoichiometric coefficient and A, B, C, D specific heat constants. A is the vapor phase average specific heat in J/mol*K (Joule per mole Kelvin). Shape must be N x 5.
      Ex) np.array([coeff1, A1, B1, C1, D1], [coeff2, A2, B2, C2, D2], [coeff3, A3, B3, C3, D3])
  T : float
    Current temperature of the reaction in K (Kelvin).
  deltaH_0 : float
    Standard heat of reaction in J/mol (Joules per mole).
  deltaG_0 : float
    Standard Gibbs free energy in J/mol (Joules per mole).
  T_0 : float
    Standard reference temperature in K (Kelvin). Most standard heat of formation tables are calcualted at 298 K (Kelvin).

  Returns:
  -----------
  deltaG : float
    Estimated Gibbs free energy of the reaction in J/mol (Joules per mole).
  '''
  phi = T / T_0
  prod = np.atleast_1d(prod).reshape(-1, 5)
  reac = np.atleast_1d(reac).reshape(-1, 5)
  delta = np.sum(np.c_[prod[:, 0]]*prod[:, 1:], axis=0) - np.sum(np.c_[reac[:, 0]]*reac[:, 1:], axis=0)
  return deltaH_0 + (deltaG_0 - deltaH_0) * phi + T_0 * delta[0] * (phi - 1. - phi*np.log(phi)) - .5 * T_0**2 * delta[1] * (phi**2 - 2.*phi + 1.) -  (1./6.) * delta[2] * T_0**3 * (phi**3 - 3.*phi + 2.) - delta[3] * ( (phi**2 - 2.*phi + 1.) / phi) / (2.*T_0) 

def gibbs_est_HandS(prod: npt.NDArray, reac: npt.NDArray, T: float, deltaH_0: float, deltaS_0: float, T_0: float = 298.) -> float:
  '''
  Estimates Gibbs free energy of a balanced chemical reaction via its standard net change in enthalpy and entropy. A negative value indicates a spontaneous reaction.
  
  Parameters:
  -----------
  prod : NDArray
    The product species' stoichiometric coefficient and A, B, C, D specific heat constants. A is the vapor phase average specific heat in J/mol*K (Joule per mole Kelvin). Shape must be N x 5.
      Ex) np.array([coeff1, A1, B1, C1, D1], [coeff2, A2, B2, C2, D2], [coeff3, A3, B3, C3, D3])
  reac : NDArray
    The reactant species' stoichiometric coefficient and A, B, C, D specific heat constants. A is the vapor phase average specific heat in J/mol*K (Joule per mole Kelvin). Shape must be N x 5.
      Ex) np.array([coeff1, A1, B1, C1, D1], [coeff2, A2, B2, C2, D2], [coeff3, A3, B3, C3, D3])
  T : float
    Current temperature of the reaction in K (Kelvin).
  deltaH_0 : float
    Standard heat of reaction in J/mol (Joules per mole).
  deltaS_0 : float
    Standard entropy of reaction in J/mol*K (Joules per mole Kelvin).
  T_0 : float
    Standard reference temperature in K (Kelvin). Most standard heat of formation tables are calcualted at 298 K (Kelvin).

  Returns:
  -----------
  deltaG : float
    Estimated Gibbs free energy of the reaction in J/mol (Joules per mole).
  '''
  prod = np.atleast_1d(prod).reshape(-1, 5)
  reac = np.atleast_1d(reac).reshape(-1, 5)
  delta = np.sum(np.c_[prod[:, 0]]*prod[:, 1:], axis=0) - np.sum(np.c_[reac[:, 0]]*reac[:, 1:], axis=0)
  H = deltaH_0 + delta[0] * (T - T_0) + delta[1] / 2. * (T**2 - T_0**2) + delta[2] / 3. * (T**3 - T_0**3)
  S = deltaS_0 + delta[0] * np.log(T/T_0) + delta[1]*(T-T_0) + delta[2] / 2. * (T**2 - T_0*2)
  return H - T * S

def k_est_gibbs(G: float, T: float) -> float:
  '''
  Estimates equilibrium constant of a reaction from Gibbs free energy.
  
  Parameters:
  -----------
  G : float
    Gibbs free energy of the reaction in J/mol (Joules per mole).
  T : float
    Current temperature of the reaction in K (Kelvin).

  Returns:
  -----------
  K : float
    Equilibrium Constant (units vary).
  '''
  return np.exp(G / (-8.314 * T))

def fate_analysis(m: float, props: list, env_vol: list, env_dens : list, env_props: list) -> npt.NDArray:
  '''
  Calculates the volumetric retention of a compound in the environment based on its chemical properties.
  
  Parameters:
  -----------
  m : float
    Mass of compound released into the environment in kg (kilograms).
  props : list
    Chemical properties of the compound being analyzed. Shape must be 1 x 7.
      Ex) np.array([Molecular Weight (g/mol), Aqueous Solubility (g/m^3), Henry's Law Constant (Pa*m^3/mol), K_ow (unitless), K_oc (unitless), Temperature (K), Liquid Vapor Pressure (Pa) ])
  env_vol : list
    Volume of each environmental retention phase in m^3 (cubic meters). Phases with a volume of 0. are ignored. Length must be 7.
      Ex) np.array([Air, Water, Soil, Bottom Sediment, Suspended Sediment, Fish, Aerosols])
  env_dens : list
    Density of each environmental retention phase in kg/m^3 (kilograms per cubic meter) except Lipid Content in Fish (unitless). Phases with a density of 0. are ignored. Length must be 4.
      Ex) np.array([Soil, Bottom Sediment, Suspended Sediment, Lipid Content in Fish])
  env_props : list
    Environmental properties of the compound (unitless). Properties with a volume of 0. are ignored. Length must be 4. 
      Ex) [Mass Fraction of Organic Carbon in Soil, Mass Fraction of Organic Carbon in Bottom Sediment, Mass Fraction of Organic Carbon in Suspended Sediment, Lipid Content of Fish]
  
  Returns
  -----------
  fate : NDArray
    Quantity of compound retained within each environmental phase in mol/m^3, mol, and kg (moles per cubic meter, moles, and kilograms). Shape is 3 x 7. 
      Ex) np.array([[Air, Water, Soil, Bottom Sediment, Suspended Sediment, Fish, Aerosols] (mol/m^3),
                    [Air, Water, Soil, Bottom Sediment, Suspended Sediment, Fish, Aerosols] (mol),
                    [Air, Water, Soil, Bottom Sediment, Suspended Sediment, Fish, Aerosols] (kg) ])
  env_cap : NDArray
    Capacity within each environmental phase in mol/Pa*m^3 (moles per Pascal meter cubed).
      Ex) np.array([Air, Water, Soil, Bottom Sediment, Suspended Sediment, Fish, Aerosols])
  fugac : float
    Fugacity of the compound in the environment in Pa (Pascals).
  '''
  env_vol = np.atleast_1d(env_vol); env_props = np.atleast_1d(env_props); props = np.atleast_1d(props)

  env_cap = np.empty(7)
  env_cap[0] = (1. / (props[5]*8.314))                              #atmos
  env_cap[1] = 1./props[2]                                          #water
  env_cap[2] = env_cap[1]*env_dens[0]*env_props[0]*(props[4]/1000)  #soil
  env_cap[3] = env_cap[1]*env_dens[1]*env_props[1]*(props[4]/1000)  #sed
  env_cap[4] = env_cap[1]*env_dens[2]*env_props[2]*(props[4]/1000)  #sus sed
  env_cap[5] = env_cap[1]*env_dens[3]*env_props[3]*(props[3]/1000)  #fish
  if props[6] == 0.:
    env_cap[6] = 0.
  else:
    env_cap[6] = env_cap[0]*6e6/props[6]                            #aerosol

  fugac = m * 1000 / props[0] / np.sum(env_cap * env_vol)
  
  fate = np.ones((3, 7))
  fate[0] = env_cap * fugac 
  fate[1] = fate[0] * env_vol
  fate[2] = fate[1] * props[0]
  return fate, env_cap, fugac

def atom_economy(atoms: npt.NDArray) -> float:
  '''
  Calculates the atom economy of a reaction for a specific produced compound by elemental mass.
  
  Parameters:
  -----------
  atoms : NDArray
    The element's molecular weight in kg/kmol (kilograms per kilomole), the element's quantity in the target product and the element's element's quantity among all reactants. Shape must be N x 3.
      Ex) For a reaction containing 4 elements: np.array([[12., 4, 0], [16, 5, 2], [1, 22, 8], [35.5, 2, 0]])

  Returns
  -----------
  aecon : float
    Atomic economy of the desired product.
  '''

  atoms = np.atleast_1d(atoms).reshape(-1, 3)
  return np.sum(atoms[:,0] * atoms[:,1]) / np.sum(atoms[:,0] * atoms[:,2])

def k_est_regression(P: float, T: float, query: str | npt.NDArray):
  df = pd.read_csv('datasets/k_hydrocarbons.csv')
  coeff = np.array([])
  if type(query) == str:
    coeff = df.loc[df.loc[:,'Hydrocarbon'] == query].iloc[:,3:14].to_numpy()
  else:
    for i, item in enumerate(query):
      coeff = np.append(coeff,df.loc[df.loc[:,'Hydrocarbon'] == item].iloc[:,3:14].to_numpy()).reshape((-1,11))
  return np.exp(coeff) # OK so we do not have a correct equation (or coefficients for all we know...) 