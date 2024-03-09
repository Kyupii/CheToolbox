import numpy as np
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
        Absolute roughness value of the pipe in m (meters). Default value of 0.10e-3 meters is the roughenss value of moderately-corroded carbon steel.

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
    
