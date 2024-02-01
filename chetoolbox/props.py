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
  g = g.reshape(-1, 2)
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

def Kow_est(f: npt.ArrayLike) -> float:
  '''
  Estimates the octanol / water equilibrium constant of a compound via the group contribution method.

  Parameters:
  -----------
  g : ArrayLike
    The frequency of a group's appearance, the group's contribution value, and the group's correction factor if necessary (0 for no correction). Shape must be N x 3.
      Ex) For a molecule containing 4 groups: np.array([[3, 1.233, 0], [1, 23.5, 0], [2, 44.6, 8.6], [7, 103.6, 13]])
  
  Returns:
  --------
  K_ow : float
    Estimated octanol / water equilibrium constant (unitless).
  '''
  return 0.229 + np.sum(f[:,0] * f[:,1]) + np.sum(f[:,0] * f[:,2])

def bioconc_est(K_ow: float, c: list = None) -> (float, str):
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
  if c == None:
    bcf = 10.**(.79 * np.log10(K_ow) - .4)
  else:
    bcf = 10.**(.77 * np.log10(K_ow) + np.sum(c) - .7)
  if bcf <= 250:    cat = "Low Potential for Tissue Accumulation"
  elif bcf <= 1000: cat = "Moderate Potential for Tissue Accumulation"
  else:             cat = "High Potential for Tissue Accumulation"
  return bcf, cat

def water_sol_est(K_ow: float, c: list = None, T_m: float = None, MW: float = None) -> (float, str): 
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
    Molecular weight in kg/kmol (kilograms per kilomole). 
  
  Returns:
  --------
  sol : float
    Estimated water solubility in mol/L (moles per liter).
  cat : float
    Qualitative solubility in water.
  '''
  if T_m == None:
    sol = 10.**(.796  - .854 * np.log10(K_ow) - .00728 * MW + np.sum(c))
  elif MW == None:
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

def henry_est(g : npt.ArrayLike, T: float = None) -> (float, str | None): 
  '''
  Estimates the Henry's Law constant of a compound by group contribution method.

  Parameters:
  -----------
  g : ArrayLike
    The frequency of a group's appearance, the group's contribution value, and the group's correction factor if necessary (0 for no correction). Shape must be N x 3.
      Ex) For a molecule containing 4 groups: np.array([[3, 1.233, 0], [1, 23.5, 0], [2, 44.6, 8.6], [7, 103.6, 13]])
  T : float
    Current temperature of the compound in K (Kelvin).

  Returns:
  -----------
  H : float
    Estimated Henry's Law constant of the compound (unitless).
  '''
  g = g.reshape(-1, 3)
  H = 10 ** -(np.sum(g[:,0] * g[:,1]) + np.sum(g[:,0] * g[:,2])) #this is unitless H
  if T == None: return H, None

  R = 8.20575e-5 # atm*m^3/mol*K
  H_units = H*R*T
  if H_units <= 1.e-7 :   cat = "Non-volatile"
  elif H_units <= 1.e-5:  cat = "Slightly Volatile"
  elif H_units <= 1.e-3:  cat = "Moderately Volatile"
  elif H_units <= 1.e-1:  cat = "Volatile"
  else:                   cat = "Very Volatile"
  return H, cat

def soil_sorb_est(g: npt.ArrayLike, mole_con: npt.ArrayLike | float) -> float:
  '''
  Estimates the soil sobrtion coefficient of a compound via the group contribution method.

  Parameters:
  -----------
  g : ArrayLike
    The frequency of a group's appearance and the group's contribution value. Shape must be N x 2.
      Ex) For a molecule containing 4 groups: np.array([[3, 1.233], [1, 23.5], [2, 44.6], [7, 103.6]])
  mole_con: npt.ArrayLike | float
    First order molecular connectivity index of the compound. Must be precomputed (float) or of shape 1 x 2N.
      Ex) np.array([1, 3, 1, 3, 2, 3, 2, 1]) or 2.68
  
  Retruns:
  -----------
  K_oc : float
    Estimated soil sobrtion coefficient in ug*mL/g*mg (the mass ratio of compound to organic carbon in soil [mg/g] divided by the concentration of the compound in water [mg/mL]).
  '''
  if type(mole_con) == npt.ArrayLike:
    mole_con = mole_con.reshape(-1, 2)
    mole_con = np.sum( (1. / (mole_con[:,0]*mole_con[:,1]) )**.5 )
  
  g = g.reshape(-1, 2)
  return .53 * mole_con + .62 + np.sum( g[:,0] * g[:,1] )

# TODO check units
def cp_est(const: list, T: float) -> float:
  '''
  Estimates the specific heat of a compound.

  Parameters:
  -----------
  const : list
    a, b, c, and d constants of the compound.
  T : float
    Current temperature of the compound in K (Kelvin)??.

  Returns:
  -----------
  Cp : float
    Estimated specific heat constant of the compound in kJ/mol (kilojoules per mole)??.
  '''
  return const[0] + const[1]*T + const[2]*T**2 + const[3] / T**2

# TODO finish this, not sure how you want it to work
def deltaH_est(prod: npt.ArrayLike, reac: npt.ArrayLike, H_0: float, T: float, T_0= 298) -> float:
  '''
  Estimates specific heat for a particular species.

  Parameters:
  -----------
  prod : ArrayLike
    array of products' stoichiometric coefficients & their corresponding a, b, c, and d constants.
      Ex)  np.array([coeff1, a, b, c, d], [coeff2, a, b, c, d], [coeff3, a, b, c, d])
  reac : ArrayLike
    array of reactants' stoichiometric coefficients & their corresponding a, b, c, and d constants.
      Ex)  np.array([coeff1, a, b, c, d], [coeff2, a, b, c, d], [coeff3, a, b, c, d])
  H_0 : float
    Standard heat of reaction for a particular reaction at T_0 in KJ/mol (Kilojoules per mole).
  T : float
    Temperature at which Gibbs free energy is to be estimated at in K (Kelvin)
  T_0 : float (optional)
    Standard reference temperature. usually 298 K (Kelvin)

  Returns:
  -----------
  deltaH : float
    KJ/mol (kilojoules per mole)
  '''
  phi = T / T_0
  delta = np.array([np.sum(prod[:,0]*prod[:,0]) - np.sum(reac[:,0]*reac[:,0]), # delta A
                    np.sum(prod[:,0]*prod[:,1]) - np.sum(reac[:,0]*reac[:,1]), # delta B
                    np.sum(prod[:,0]*prod[:,2]) - np.sum(reac[:,0]*reac[:,2]), # delta C
                    np.sum(prod[:,0]*prod[:,3]) - np.sum(reac[:,0]*reac[:,3])]) # delta D
  return H_0 + delta[0] * (T - T_0) + delta[1] / 2 * (T**2 - T_0**2) + delta[3] / 3 *(T**3 - T_0**3) + delta[3] / T_0 * ((phi - 1) / phi)

def gibbs_est_cp(prod: npt.ArrayLike, reac: npt.ArrayLike, H_0: float, G_0: float,T: float, T_0:float = 298) -> float:
  '''
  Estimates the Gibbs free energy for a particular reaction.

  Parameters:
  -----------
  prod : ArrayLike
    array of products' stoichiometric coefficients & their corresponding a, b, c, and d constants.
      Ex)  np.array([coeff1, a, b, c, d], [coeff2, a, b, c, d], [coeff3, a, b, c, d])
  reac : ArrayLike
    array of reactants' stoichiometric coefficients & their corresponding a, b, c, and d constants.
      Ex)  np.array([coeff1, a, b, c, d], [coeff2, a, b, c, d], [coeff3, a, b, c, d])
  H_0 : float
    Standard heat of reaction for a particular reaction at T_0 in KJ/mol (Kilojoules per mole).
  G_0 : float
    Standard Gibbs free energy for a particular reaction at T_0 in KJ/mol (Kilojoules per mole).
  T : float
    Temperature at which Gibbs free energy is to be estimated at in K (Kelvin)
  T_0 : float (optional)
    Standard reference temperature. usually 298 K (Kelvin)

  Returns:
  -----------
  deltaG : float
    KJ/mol (kilojoules per mole)
  '''
  phi = T / T_0
  delta = np.array([np.sum(prod[:,0]*prod[:,0]) - np.sum(reac[:,0]*reac[:,0]), # delta A
                    np.sum(prod[:,0]*prod[:,1]) - np.sum(reac[:,0]*reac[:,1]), # delta B
                    np.sum(prod[:,0]*prod[:,2]) - np.sum(reac[:,0]*reac[:,2]), # delta C
                    np.sum(prod[:,0]*prod[:,3]) - np.sum(reac[:,0]*reac[:,3])]) # delta D
  return H_0 + (G_0 - H_0) * phi + T_0 * (delta[0]) * (phi - 1 + phi*np.log(phi)) - (T_0**2) / 2 * delta[1] * (phi**2 - 2*phi + 1) - (T_0**2)*delta[2] / 6 * (phi**3 - 3*phi + 2) - delta[3] / (2*T_0) * ( (phi**2 + 2*phi +1)/phi)

# TODO : Not totally happy with this name....
def gibbs_est_changes(prod: npt.ArrayLike, reac: npt.ArrayLike, H_0, S_0, T, T_0 = 298):
  '''
  Estimate gibbs free energy from enthalpy and entropy changes.

  Parameters:
  -----------
  prod : ArrayLike
    array of products' stoichiometric coefficients & their corresponding a, b, c, and d constants.
      Ex)  np.array([coeff1, a, b, c, d], [coeff2, a, b, c, d], [coeff3, a, b, c, d])
  reac : ArrayLike
    array of reactants' stoichiometric coefficients & their corresponding a, b, c, and d constants.
      Ex)  np.array([coeff1, a, b, c, d], [coeff2, a, b, c, d], [coeff3, a, b, c, d])
  H_0 : float
    Standard change of enthalpy at T_0 in KJ/mol (kilojoules per mole)
  S_0 : float
    Standard change of entropy at T_0 in KJ/mol (kilojoules per mole)
  T : float
    Temperature at which Gibbs free energy is to be estimated at in K (Kelvin)
  T_0 : float (optional)
    Standard reference temperature. usually 298 K (Kelvin)

  Returns:
  -----------
  G : float
    Gibbs free energy in KJ/mol (Kilojoules per mole)
  '''
  delta = np.array([np.sum(prod[:,0]*prod[:,0]) - np.sum(reac[:,0]*reac[:,0]), # delta A
            np.sum(prod[:,0]*prod[:,1]) - np.sum(reac[:,0]*reac[:,1]), # delta B
            np.sum(prod[:,0]*prod[:,2]) - np.sum(reac[:,0]*reac[:,2])]) # delta C
  H = H_0 + delta[0] * (T - T_0) + delta[1] / 2 * (T**2 - T_0**2) + delta[2] / 3 * (T**3 - T_0**3)
  S = S_0 + delta[0] * np.log(T/T_0) + delta[1]*(T-T_0) + delta[2] / 2 * (T**2 - T_0*2)
  return H - T * S

def k_est(G,T):
  '''
  Estimates equilibrium constant from Gibbs free energy.

  Parameters:
  -----------
  prod : ArrayLike
    array of products' stoichiometric coefficients & their corresponding a, b, c, and d constants.
      Ex)  np.array([coeff1, a, b, c, d], [coeff2, a, b, c, d], [coeff3, a, b, c, d])
  reac : ArrayLike
    array of reactants' stoichiometric coefficients & their corresponding a, b, c, and d constants.
      Ex)  np.array([coeff1, a, b, c, d], [coeff2, a, b, c, d], [coeff3, a, b, c, d])
  G : float
    Gibbs free energy the temperature K is to be estimated in KJ/mol (kilojoules per mole)

  Returns:
  -----------
  K : float
    Equilibrium Constant (Units vary)
  '''
  return (-G) / (8.314 * T)