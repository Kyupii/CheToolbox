import numpy as np
from scipy.optimize import root
# Calculate the pressure per unit length 
# Inputs and outputs are in Metric
#
def PressureDrop(Q:float, mu:float, D:float, rho:float, epsilon : float = 0.10e-3, units : str= 'SI') -> float: 
    r""" solve for the pressure drop of a straight pipe using DarcyWeisbach equation & Colebrook White equation
    Parameters
    ----------
    Q : float
        A float value of the volumetric flow rate in $m^3/s$
    mu : float
        a float value of the viscosity of the fluid in pascal * seconds
    D : float
        The diameter of the pipe in meters
    epsilon : float
        the absolute roughness value of the pipe in meters. It is .10e-3 m by default as the roughenss value of moderately corroded carbon steel pipe
    units : str
        an option to change the units of the function. it is SI by default.

    Returns
    ----------
    PressureDrop : float
        The amount of pressure drop per unit length. by default it is in Pascals/meter
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
    

