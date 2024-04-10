import numpy as np
import numpy.typing as npt
from typing import Callable
from matplotlib import pyplot as plt
import common, props

from scipy.optimize import root
def PressureDrop(Q: float, mu: float, D: float, rho: float, epsilon: float = 0.10e-3) -> float: 
  r"""
  Solve for the pressure drop of a straight pipe using the Darcy-Weisbach equation & Colebrook White equation.
  
  Parameters
  ----------
  Q : float
    Volumetric flowrate in m^3/s (meters cubed per second).
  mu : float
    Viscosity of the fluid in Pa*s (pascal seconds).
  D : float
    Pipe diameter in m (meters).
  mu : float
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

def free_mean_path(T_and_P: npt.NDArray, d_molecule: npt.NDArray | float):
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
  mfp : NDArray | float
    Mean free path of the molecule in m (meters).
  '''
  T_and_P = np.atleast_2d(T_and_P)
  kB = 1.380649e-23 # J/K*mol
  return kB * T_and_P[:, 0] / (np.sqrt(2.) * T_and_P[:, 1] * np.pi * d_molecule**2)

def diffuse_knudsen(MW: npt.NDArray, r_pore: float, T: float, porous: float | None = None, tort: float | None = None):
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
    Effective knudsen diffusivity in cm**2/s (squared centimeters per second).
  '''
  D = 9700. * r_pore * (T / MW)**.5
  if porous is not None and tort is not None:
    return D * porous / tort
  return .25 * D
