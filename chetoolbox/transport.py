import numpy as np
import numpy.typing as npt
from typing import Callable
from matplotlib import pyplot as plt
from . import common, props

from scipy.optimize import root
def PressureDrop(Q: float, mu: float, D: float, rho: float, epsilon: float = 0.10e-3) -> float: 
  """
  Solve for the pressure drop of a straight pipe using the Darcy-Weisbach equation & Colebrook White equation.
  
  Parameters
  ----------
  Q : float
    Volumetric flowrate in m^3/s (meters cubed per second).
  mu : float
    Viscosity of the fluid in Pa*s (pascal seconds).
  D : float
    Pipe diameter in m (meters).
  rho : float
    Density of the fluid in kg/m^3 (kilograms per meter cubed).
  epsilon : float
    Absolute roughness value of the pipe in m (meters). Default value of 0.10e-3 meters is the roughness value of moderately-corroded carbon steel.

  Returns
  ----------
  PressureDrop : float
    Pressure drop per unit length in Pa/m (pascals per meter).
  """
  v = Q / (np.pi * ((D/2)**2))
  Re = rho * v * D / mu
  def ColebrookWhite(v):
    f = v[0] # Unpack a vector of unknowns
    return [ (1 / np.sqrt(f)) - 
            (-2 * np.log10((epsilon/(3.7 * D)) + 2.51/(Re*np.sqrt(f))))] #subtract RHS from LHS. If each side is equivalent, we expect 0
  if Re<4000:
    f = 64 / Re 
  else:
    f = root(ColebrookWhite,[0.03]).x[0]
  return -(f * (rho / 2) * (v**2 /D))

def mean_free_path(T_and_P: npt.NDArray, d_molecule: npt.NDArray | float) -> npt.NDArray | float:
  '''
  Calculates the mean free path of molecules in unrestricted diffusion.  
  
  Parameters
  ----------
  T_and_P : NDArray
    Temperature in K (Kelvin) and pressure in Pa (Pascals) pairs.
  d_molecule : NDArray | float
    Diameter of a molecule in m (meters).
  
  Returns
  ----------
  lamda : NDArray | float
    Mean free path of the molecule in m (meters).
  '''
  T_and_P = np.atleast_2d(T_and_P)
  kB = 1.380649e-23 # J/K*mol
  return kB * T_and_P[:, 0] / (np.sqrt(2.) * T_and_P[:, 1] * np.pi * d_molecule**2)

def mean_free_speed(T: npt.NDArray | float, MW: npt.NDArray | float,) -> npt.NDArray | float:
  '''
  Calculates the mean free speed of molecules in unrestricted diffusion. If multiple temperatures (T) and molecular weights (MW) are input, the resulting array will be of size T x MW.
  
  Parameters
  ----------
  T : NDArray ] float
    Temperature in K (Kelvin).
  MW : NDArray | float
    Molecular weight of the compound.
  
  Returns
  ----------
  mfs : NDArray | float
    Mean free speed of the molecule in m/s (meters per second).
  '''
  T = np.atleast_1d(T).reshape(-1, 1)
  kB = 1.380649e-23 # J/K*mol
  NA = 6.02214076e23 # particles/mol
  mfs = 8 * kB * T * NA / (np.pi * MW)
  return mfs[0] if mfs.size == 1 else mfs

def reynolds(L: npt.NDArray | float, flowspeed: npt.NDArray | float, rho: float, mu: float) -> npt.NDArray | float:
  '''
  Calculates the Reynolds number of a dynamic fluid. If multiple characteristic lengths (L) and flow speeds (FS) are input, the resulting array will be of size FS x L. Units must cancel.
  
  Parameters
  ----------
  L : NDArray | float
    Characteristic length of the system.
  flowspeed : NDArray | float
    Linear flow speed of the fluid, typically perpendicular to the characteristic length.
  rho : float
    Density of the fluid.
  mu : float
    Dynamic viscosity of the fluid.
  
  Returns
  ----------
  Re : NDArray | float
    Reynolds number of the dynamic fluid (Unitless).
  '''
  L = np.atleast_1d(L)
  flowspeed = np.atleast_1d(flowspeed).reshape(-1, 1)
  Re = rho * L * flowspeed / mu
  return Re[0] if Re.size == 1 else Re

def reynolds_pipe(din: npt.NDArray | float, Q_dot: npt.NDArray | float, rho: float, mu: float) -> npt.NDArray | float:
  '''
  Calculates the Reynolds number of a dynamic fluid in a circular pipe. If multiple inner diameters (D) and flow rates (Q) are input, the resulting array will be of size Q x D. Units must cancel.
  
  Parameters
  ----------
  din : NDArray | float
    Inner diameter of the pipe.
  Q_dot : NDArray | float
    Volumetric flowrate of the fluid through the pipe. If rho set == 1., mass flowrate (W_dot) may be used.
  rho : float
    Density of the fluid.
  mu : float
    Dynamic viscosity of the fluid.
  
  Returns
  ----------
  Re : NDArray | float
    Reynolds number of the dynamic fluid (Unitless).
  '''
  din = np.atleast_1d(din)
  Q_dot = np.atleast_1d(Q_dot).reshape(-1, 1)
  A = np.pi * din**2 / 4.
  Re = rho * Q_dot * din / (mu * A)
  return Re[0] if Re.size == 1 else Re

def schmidt(massDiff: npt.NDArray | float, rho: float, mu: float) -> npt.NDArray | float:
  '''
  Calculates the Schmidt number of a dynamic fluid. Units must cancel.
  
  Parameters
  ----------
  massDiff : NDArray | float
    Rate of mass diffusion through the fluid.
  rho : float
    Density of the fluid.
  mu : float
    Dynamic viscosity of the fluid.
  
  Returns
  ----------
  Sc : NDArray | float
    Schmidt number of the dynamic fluid (Unitless).
  '''
  return mu / (rho * massDiff)

def sherwood(massDiff: npt.NDArray | float, L: npt.NDArray | float, h: float) -> npt.NDArray | float:
  '''
  Calculates the Sherwood number of a dynamic fluid. If multiple characteristic lengths (L) and mass diffusions (MD) are input, the resulting array will be of size MD x L. Units must cancel.
  
  Parameters
  ----------
  massDiff : NDArray | float
    Rate of mass diffusion through the fluid.
  L : NDArray | float
    Characteristic length of the system.
  h : float
    Convective mass transfer film coefficient.
  
  Returns
  ----------
  Sh : NDArray | float
    Sherwood number of the dynamic fluid (Unitless).
  '''
  L = np.atleast_1d(L)
  massDiff = np.atleast_1d(massDiff).reshape(-1, 1)
  Sh = h * L / massDiff
  return Sh[0] if Sh.size == 1 else Sh

def nusselt(thermCond: npt.NDArray | float, L: npt.NDArray | float, h: float) -> npt.NDArray | float:
  '''
  Calculates the Nusselt number for a dynamic fluid. If multiple characteristic lengths (L) and thermal conductivity (TC) are input, the resulting array will be of size TC x L. Units must cancel.
  
  Parameters
  ----------
  thermCond : NDArray | float
    Thermal conductivity of the fluid.
  L : NDArray | float
    Characteristic length of the system.
  h : float
    Convective heat transfer coefficient.
  
  Returns
  ----------
  Nu : NDArray | float
    Nusselt number of the dynamic fluid (Unitless).
  '''
  L = np.atleast_1d(L)
  thermCond = np.atleast_1d(thermCond).reshape(-1, 1)
  Nu = h * L / thermCond
  return Nu[0] if Nu.size == 1 else Nu

def knudsen(mean_free_path: npt.NDArray | float, L: npt.NDArray | float, h: float) -> npt.NDArray | float:
  '''
  Calculates the Knudsen number for a dynamic fluid. If multiple characteristic lengths (L) and mean free path (MFP) are input, the resulting array will be of size MFP x L. Units must cancel.
  
  Parameters
  ----------
  mean_free_path : NDArray | float
    Mean free path of the dynamic fluid particles.
  L : NDArray | float
    Characteristic length of the system.
  
  Returns
  ----------
  Kn : NDArray | float
    Knudsen number of the dynamic fluid (Unitless).
  '''
  L = np.atleast_1d(L)
  mean_free_path = np.atleast_1d(mean_free_path).reshape(-1, 1)
  Kn = mean_free_path / L
  return Kn[0] if Kn.size == 1 else Kn

def collision_integral(T: npt.NDArray | float, eps_geo: float) -> npt.NDArray | float:
  '''
  Calculates the collision integral of molecules in a binary mixture.
  
  Parameters
  ----------
  T : npt.NDArray | float
    Environment temperature in K (Kelvin).
  eps_geo : float
    Geometric average of molecular interaction coefficients.
  
  Returns
  ----------
  omegaD : NDArray | float
    Collision integral of the binary mixture interaction.
  '''
  a = 1.06036; b = .1561
  c = .193; d = .47635
  e = 1.03587; f = 1.52996
  g = 1.76474; h = 3.89411
  x = T / eps_geo
  return (a / x**b) + (c / np.e**(d*x)) + (e / np.e**(f*x)) + (g / np.e**(h*x))

# TODO: double check the units on the theoretical binary gas mixture, if ever stated
def diffuse_binary_mixture_theory(MW: npt.NDArray, collis_diam: npt.NDArray, epsilon: npt.NDArray, T: float, P: float, molar_volume_B: float | None = None, boiling_temp_B: float | None = None):
  '''
  Calculates the diffusion of a compound in binary mixture, according to a theoretical estimation (not to be used for real-world applications). The first compound's properties should be A, the minority (solute) compound, while the second compound's properties should be B, the majority (solvent) compound.
  
  Parameters
  ----------
  MW : NDArray
    Molecular weight of the minority and majority (solute and solvent) compounds in g/mol (grams per mole). Must be of length 2.
  collis_diam : NDArray
    Collision diameter of the minority and majority (solute and solvent) compounds in Å (angstroms). Must be of length 2.
  epsilon : NDArray
    Molecular weight of the minority and majority (solute and solvent) compounds in g/mol (grams per mole). Must be of length 2.
  T : float
    Environment temperature in K (Kelvin).
  P : float
    Environment pressure in bar (bar).
  molar_volume_B : float (Optional)
    Molar volume of B in Å^3 (cubic angstroms). Used for an estimate of the overall collision diameter if a compound's collision diameter is unknown.
  boiling_temp_B : float (Optional)
    Boiling temperature of B at the given pressure in K (Kelvin). Used for an estimate of the overall molecular interaction if a compound's molecular interaction is unknown.
  
  Returns
  ----------
  Dab : NDArray | float
    Diffusivity of the minority compound through the majority compound in cm^2/s (square centimeters per second).
  '''
  MW = np.atleast_1d(MW)
  collis_diam = np.atleast_1d(collis_diam)
  epsilon = np.atleast_1d(epsilon)
  
  MWab = 2. * MW.prod() / MW.sum()
  Oab = collis_diam.sum() / 2.
  if molar_volume_B is not None:
    Oab = 1.18 * molar_volume_B ** (1./3.)
  Eab = common.geomean(epsilon)
  if boiling_temp_B is not None:
    Eab = 1.15 * boiling_temp_B
  omegaD = collision_integral(T, Eab)
  
  return .0026 * T**(3./2.) / (P * np.sqrt(MWab) * np.square(Oab) * omegaD)

def diffuse_wilke_lee(MW: npt.NDArray, collis_diam: npt.NDArray, epsilon: npt.NDArray, T: float, P: float, molar_volume_B: float | None = None, boiling_temp_B: float | None = None):
  '''
  Calculates the diffusion of a compound in binary gas mixture using Wilke & Lee's estimation. The first compound's properties should be A, the minority (solute) compound, while the second compound's properties should be B, the majority (solvent) compound.
  
  Parameters
  ----------
  MW : NDArray
    Molecular weight of the minority and majority (solute and solvent) compounds in g/mol (grams per mole). Must be of length 2.
  collis_diam : NDArray
    Collision diameter of the minority and majority (solute and solvent) compounds in Å (angstroms). Must be of length 2.
  epsilon : NDArray
    Molecular weight of the minority and majority (solute and solvent) compounds in g/mol (grams per mole). Must be of length 2.
  T : float
    Environment temperature in K (Kelvin).
  P : float
    Environment pressure in bar (bar).
  molar_volume_B : float (Optional)
    Molar volume of B in Å^3 (cubic angstroms). Used for an estimate of the overall collision diameter if a compound's collision diameter is unknown.
  boiling_temp_B : float (Optional)
    Boiling temperature of B at the given pressure in K (Kelvin). Used for an estimate of the overall molecular interaction if a compound's molecular interaction is unknown.
  
  Returns
  ----------
  Dab : NDArray | float
    Diffusivity of the minority compound through the majority compound in cm^2/s (square centimeters per second).
  '''
  MW = np.atleast_1d(MW)
  collis_diam = np.atleast_1d(collis_diam)
  epsilon = np.atleast_1d(epsilon)
  
  MWab = 2. * MW.prod() / MW.sum()
  Oab = collis_diam.sum() / 2.
  if molar_volume_B is not None:
    Oab = 1.18 * molar_volume_B ** (1./3.)
  Eab = common.geomean(epsilon)
  if boiling_temp_B is not None:
    Eab = 1.15 * boiling_temp_B
  omegaD = collision_integral(T, Eab)
  coeff = (3.03 - .98 / np.sqrt(MWab)) * 10e-3
  
  return coeff * T**(3./2.) / (P * np.sqrt(MWab) * np.square(Oab) * omegaD)

# TODO
# diffuse_einstein_stokes
# diffuse_wilke_chang
# diffuse_hayduk_aqueous
# diffuse_hayduk_organic
# diffuse_nernst_haskell

def diffuse_knudsen_united(MW: npt.NDArray, r_pore: float, T: float, porous: float | None = None, tort: float | None = None) -> npt.NDArray | float:
  '''
  Calculates the knudsen diffusivity of molecules in through porous membranes.
  
  Parameters
  ----------
  MW : NDArray | float
    Molecular weight in g/mol (grams per mole) of each molecule.
  r_pore : float
    Average pore radius in cm (centimeters).
  T : float
    Environment temperature in K (Kelvin).
  porous : float | None
    Membrane porosity, bounded [0., 1.].
  tort : float | None
    Membrane porosity, bounded [1., inf).
  
  Returns
  ----------
  D_eff : NDArray | float
    Effective knudsen diffusivity in cm^2/s (squared centimeters per second).
  '''
  D = 9700. * r_pore * (T / MW)**.5
  if porous is not None and tort is not None:
    return D * porous / tort
  return .25 * D
