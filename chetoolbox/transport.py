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

# TODO
# knudsen == mean_free_path / pore diam

# TODO
# diffuse_binary_mixture_theory
# diffuse_wilke_lee
# diffuse_einstein_stokes
# diffuse_wilke_chang
# diffuse_hayduk_aqueous
# diffuse_hayduk_organic
# diffuse_nernst_haskell

def diffuse_knudsen(MW: npt.NDArray, r_pore: float, T: float, porous: float | None = None, tort: float | None = None) -> npt.NDArray | float:
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
