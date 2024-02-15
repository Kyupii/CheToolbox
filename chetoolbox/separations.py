import numpy as np
import numpy.typing as npt
from chetoolbox import common

def raoult_XtoY(x: list, K: list) -> tuple[npt.ArrayLike, float]:
  '''
  Calculates the vapor mole fraction of a multi-component mixed phase feed (assuming liquid and gas ideality).

  Parameters
  ---------
  x : list
    Component mole fractions of the feed's liquid phase (unitless). Must sum to 1.
  K : list 
    Equalibrium constant for each component at a specific temperature and pressure (units vary). Length must equal x.
  
  Returns
  ---------
  y : ArrayLike
    Component mole fractions of the feed's vapor phase (unitless).
  error : float
    Error of calculated vapor phase component mole fractions.
  '''
  x = np.atleast_1d(x)
  K = np.atleast_1d(K)
  y = np.c_[x] * K
  error = np.sum(y) - 1
  return y, error

def raoult_YtoX(y: list, K: list) -> tuple[npt.ArrayLike, float]:
  '''
  Calculates the liquid mole fraction of a multi-component mixed phase feed (assuming liquid and gas ideality).

  Parameters
  ---------
  y : list
    Component mole fractions of the feed's vapor phase (unitless). Must sum to 1.
  K : list 
    Equalibrium constant for each component at a specific temperature and pressure (units vary). Length must equal y.
    
  Returns
  ---------
  x : ArrayLike
    Component mole fractions of the feed's liquid phase (unitless).
  error : float
    Error of calculated liquid phase component mole fractions.
  '''
  y = np.atleast_1d(y)
  K = np.atleast_1d(K)
  x = np.c_[y] / K
  error = np.sum(x) - 1
  return x, error

def psi_solver(x: list, K: list, psi: float, tol: float = 0.01) -> tuple[float, npt.ArrayLike, npt.ArrayLike, float, int]:
  '''
  Iteratively solves for the vapor/liquid output feed ratio psi (Ψ) of a multi-component fluid stream.  
  
  Parameters
  ----------
  x : list
    Component mole fractions of the liquid input feed (unitless). Must sum to 1.
  K : list
    Equalibrium constant for each component at specific temperature and pressure (units vary). Length must equal x.
  psi : float
    Initial value of psi to begin iterating on (unitless). Must be 0 <= psi <= 1.
  tol : float
    Largest error value to stop iterating and return.

  Returns
  ----------
  psi : float
    Converged vapor/liquid output feed ratio psi (Ψ) (unitless).
  x_out : ArrayLike
    Component mole fractions of the output liquid stream (unitless). 
  y_out : ArrayLike
    Component mole fractions of the output vapor stream (unitless).
  error : float
    Error at the final iteration.
  i : int
    Number of iterations to calculate to the specified tolerance.
  '''
  x = np.atleast_1d(x)
  K = np.atleast_1d(K)
  def f(psi):
    return np.sum( (x * (1 - K)) / (1 + psi * (K - 1)) )
  def f_prime(psi):
    return np.sum( (x * (1 - K)**2) / (1 + psi * (K - 1))**2)
  def psi_next(psi):
    return psi - (f(psi) / f_prime(psi))
  def error(psi):
    return np.abs(f(psi))
  
  i = 0
  while error(psi) > tol:
    psi = psi_next(psi)
    i += 1
  x_out = x / (1 + psi * (K - 1))
  y_out = (x * K) / (1 + psi * (K - 1))
  return psi, x_out, y_out, error(psi), i

def bubble_point(x: list, ant_coeff: npt.ArrayLike, P: float, tol: float = .05) -> tuple[float, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, float, int]:
  '''
  Iteratively solves for the bubble point temperature of a multi-component liquid mixture.

  Parameters
  ----------
  x : list
    Component mole fractions of the liquid mixture (unitless). Must sum to 1.
  ant_coeff : ArrayLike
    Components' coefficients for the Antoine Equation of State (unitless). Shape must be N x 3.
  P : float
    Ambient pressure of the liquid mixture in mmHg (millimeters of mercury).
  tol : float
    Largest error value to stop iterating and return.

  Returns
  ----------
  bubbleT : float
    Temperature of the liquid mixture's bubble point in C (Celcius).
  Pvap : ArrayLike
    Vapor pressure for each component at the bubble point temperature in mmHg (millimeters of mercury). 
  K : ArrayLike
    Equalibrium constant for each component at the stated pressure and bubble point temperature (units vary). 
  y : ArrayLike
    Component mole fractions of the first bubble of vapor (unitless).
  error : float
    Error at the final iteration.
  i : int
    Number of iterations to calculate to the specified tolerance.
  '''
  x = np.atleast_1d(x)
  ant_coeff = np.atleast_1d(ant_coeff).reshape(-1, 3)
  boil_points = common.antoine_T(ant_coeff, P)

  def TtoY(T):
    Pvap = common.antoine_P(ant_coeff, T)
    k = Pvap / P
    y = np.c_[x] * k
    return Pvap, k, y
  
  def err(T):
    _, _, y = TtoY(T)
    return np.sum(y, axis=0) - 1.
  
  bubbleT, error, i = common.iter(err, [np.max(boil_points), np.min(boil_points)], tol)

  Pvap, k, y = TtoY(bubbleT)
  return bubbleT, Pvap, k, y, error, i

def dew_point(y: list, ant_coeff: npt.ArrayLike, P: float, tol: float = .05) -> tuple[float, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, float, int]:
  '''
  Iteratively solves for the dew point temperature of a multi-component vapor mixture.

  Parameters
  ----------
  y : list
    Component mole fractions of the vapor mixture (unitless). Must sum to 1.
  ant_coeff : ArrayLike
    Components' coefficients for the Antoine Equation of State (unitless). Shape must be N x 3.
  P : float
    Ambient pressure of the vapor mixture in mmHg (millimeters of mercury).
  tol : float
    Largest error value to stop iterating and return.

  Returns
  ----------
  dewT : float
    Temperature of the vapor mixture's dew point in C (Celcius).
  Pvap : ArrayLike
    Vapor pressure for each component at the dew point temperature in mmHg (millimeters of mercury). 
  K : ArrayLike
    Equalibrium constant for each component at the stated pressure and dew point temperature (units vary). 
  x : ArrayLike
    Component mole fractions of the first dew of liquid (unitless).
  error : float
    Error at the final iteration.
  i : int
    Number of iterations to calculate to the specified tolerance.
  '''
  y = np.atleast_1d(y)
  ant_coeff = np.atleast_1d(ant_coeff).reshape(-1, 3)
  boil_points = common.antoine_T(ant_coeff, P)

  def TtoX(T):
    Pvap = common.antoine_P(ant_coeff, T)
    k = Pvap / P
    x = np.c_[y] / k
    return Pvap, k, x
  
  def err(T):
    _, _, x = TtoX(T)
    return np.sum(x, axis=0) - 1.
  
  dewT, error, i = common.iter(err, [np.max(boil_points), np.min(boil_points)], tol)
  
  Pvap, k, y = TtoX(dewT)
  return dewT, Pvap, k, y, error, i

def liq_frac_subcooled(Cpl: float, heatvap: float, Tf: float, Tb: float) -> float:
  '''
  Calculates the liquid fraction of a subcooled bianary liquid mixture feed.

  Parameters:
  -----------
  Cpl : float
    Specific heat of the liquid feed in kJ/mol*K (kilojoules per mole Kelvin).
  heatvap : float
    Heat of vaporization of the liquid feed in kJ/mol (kilojoules per mole).
  Tf : float
    Temperature of the liquid feed in K (Kelvin).
  Tb : float
    Bubble temperature of the liquid feed in K (Kelvin).

  Returns:
  -----------
  q : float
    Liquid fraction of the feed (unitless).
  '''
  return 1. + Cpl * (Tb - Tf) / heatvap

def liq_frac_superheated(Cpv: float, heatvap: float, Tf: float, Td: float) -> float:
  '''
  Calculates the liquid fraction of a superheated bianary vapor mixture feed.

  Parameters:
  -----------
  Cpv : float
    Specific heat of the vapor feed in kJ/mol*K (kilojoules per mole Kelvin).
  heatvap : float
    Heat of vaporization of the vapor feed in kJ/mol (kilojoules per mole).
  Tf : float
    Temperature of the vapor feed in K (Kelvin).
  Td : float
    Dew temperature of the vapor feed in K (Kelvin).

  Returns:
  -----------
  q : float
    Liquid fraction of the feed (unitless).
  '''
  return -Cpv * (Tf - Td) / heatvap

def eq_curve_estim(points: npt.ArrayLike, alpha: float = None) -> common.EqualibEq:
  '''
  Estimates an equalibrium curve for a bianary mixture. Assumes constant equalibrium ratio (K1 / K2) between the two species.

  Parameters:
  -----------
  points : ArrayLike
    Points on an equalibrium curve. Bounded (0, 1). Shape must be N x 2.
      Ex) np.array([ [.2, .1], [.3, .23], [.4, .5] ])
  alpha : float (Optional)
    Relative volatility of the two species equalibrium constants (K) (unitless). Takes priority over point-based estimation.

  Returns:
  -----------
  equalibrium_curve : EqualibEq
    Equation for an equalibrium curve of a bianary mixture.
  '''
  points = np.atleast_1d(points).reshape((-1, 2))
  if alpha == None:
    alpha = np.average( points[:, 1] * (1. - points[:, 0]) / points[:, 0] * (1. - points[:, 1]) )

  return common.EqualibEq(alpha)

def mccabe_thiel_feedline(q: float, xf: float) -> common.LinearEq:
  '''
  Calculates the feed line on a McCabe Thiel Diagram for a bianary mixture distilation column. Assumes equal molar heats of vaporization.

  Parameters:
  -----------
  q : float
    Feed liquid fraction (unitless).
  xf : float
    Liquid fraction of the feed's lower boiling boint species (unitless).
  
  Returns:
  -----------
  feedline : LinearEq
    Feed line of a McCabe Thiel Diagram.
  '''
  if q == 1:
    feedline = common.vertical_line(xf)
  else:
    m = -q / (1. - q)
    y_int = m*xf + xf
    feedline = common.LinearEq(m, y_int)
  return feedline

def mccabe_thiel_otherlines(feedline: common.LinearEq, eq_feedpoint: tuple, xd: float, xb: float, Rmin_mult: float = 1.2) -> tuple[common.LinearEq, common.LinearEq, tuple[float, float], float]:
  '''
  Calculates the rectifying and stripping operating lines of a McCabe Thiel Diagram for a bianary mixture distilation column. Assumes equal molar heats of vaporization.

  Parameters:
  -----------
  feedline : LinearEq
    Feed line of a McCabe Thiel Diagram.
  eq_feedpoint : tuple
    Point of intersection between the feed line and the equalibrium line on a McCabe Thiel Diagram (unitless, unitless). Bounded [0, 1]. Length must equal 2.
  xd : float
    Liquid fraction of the distilate's lower boiling boint species (unitless).
  xb : float
    Liquid fraction of the bottoms' lower boiling boint species (unitless).
  Rmin_mult : float
    Factor by which to excede the minimum reflux ratio, Rmin (unitless). Typical reflux ratios are between 1.05 and 1.3 times Rmin. Bounded (1, inf).

  Returns:
  -----------
  rectifyline : LinearEq
    Rectifying section operating line of a McCabe Thiel Diagram.
  stripline : LinearEq
    Stripping section operating line of a McCabe Thiel Diagram.
  feedpoint : tuple
    Point of intersection between the feed line and the equalibrium line of a McCabe Thiel Diagram (unitless, unitless).
  Rmin : float
    Minimum reflux ratio of the rectifying section (unitless).
  R : float
    Reflux ratio of the rectifying section (unitless).
  '''
  # "distilate to feed at equalibrium" line
  eq_rectifyline = common.point_slope(eq_feedpoint, (xd, xd))

  # "distilate to feedpoint" line
  Rmin = eq_rectifyline.m / (1. - eq_rectifyline.m)
  R = Rmin_mult * Rmin
  m = R / (1. + R)
  y_int = xd / (1. + R)
  rectifyline = common.LinearEq(m, y_int)

  # feedpoint
  feedpoint = common.linear_intersect(feedline, rectifyline)

  # bottoms to feed point
  stripline = common.point_slope(feedpoint, (xb, xb))

  return rectifyline, stripline, feedpoint, Rmin, R

def mccabe_thiel_full_est(eq_curve: common.EqualibEq, q: float, xf: float, xd: float, xb: float, Rmin_mult: float = 1.2, tol: float = .00001):
  '''
  Calculates the reflux ratio and ideal stages of a bianary mixture distilation column, as well as thier ideal minimums. Uses a McCabe Thiel Diagram and assumes equal molar heats of vaporization.

  Parameters:
  -----------
  equalibrium_curve : EqualibEq
    Equation for an equalibrium curve of a bianary mixture.
  q : float
    Feed liquid fraction (unitless).
  xf : float
    Liquid fraction of the feed's lower boiling boint species (unitless).
  xd : float
    Liquid fraction of the distilate's lower boiling boint species (unitless).
  xb : float
    Liquid fraction of the bottoms' lower boiling boint species (unitless).
  Rmin_mult : float
    Factor by which to excede the minimum reflux ratio, Rmin (unitless). Typical reflux ratios are between 1.05 and 1.3 times Rmin. Bounded (1, inf).
  tol : float
    Largest error value to stop iterating and return.

  Returns:
  -----------
  Rmin : float
    Minimum reflux ratio of the rectifying section (unitless).
  R : float
    Reflux ratio of the rectifying section (unitless).
  min_stages : float
    Minimum number of ideal stages (includes reboiler and partial condensor if applicable).
  ideal_stages : float
    Number of ideal stages (includes reboiler and partial condensor if applicable).
  '''
  feedline = mccabe_thiel_feedline(q, xf)

  def err(x):
    return eq_curve.eval(x) - feedline.eval(x)

  x, error, i = common.iter(err, [xb, xd], tol)

  eq_feedpoint = (x, eq_curve.eval(x))

  rectifyline, stripline, feedpoint, Rmin, R = mccabe_thiel_otherlines(feedline, eq_feedpoint, xd, xb, Rmin_mult)

  def equalibrium_line_walker(y_piecewise: function, xd: float):
    y = xd
    x = eq_curve.inv(xd)
    i = 1
    while x > xb:
      y = y_piecewise(x)
      x = eq_curve.inv(y)
      i += 1
    xprev = stripline.inv(y)
    return (i - 1.) + (xprev - xb) / (xprev - x)

  def y_reflect(x):
    return x

  def y_operlines(x):
    if x >= feedpoint[0]:
      y = rectifyline.eval(x)
    else:
      y = stripline.eval(x)
    return y
  
  min_stages = equalibrium_line_walker(y_reflect, xd)
  ideal_stages = equalibrium_line_walker(y_operlines, xd)

  return Rmin, R, min_stages, ideal_stages

def distilation_stream_split(F: float, xf: float, xd: float, xb: float, R: float = None, q: float = None) -> tuple[float, float, float | None, float | None, float | None, float | None]:
  '''
  Calculates the distilate and bottom flow rates out of a bianary mixture distilation column. Optionally calculates the internal flows between the feed tray, rectifying, and stripping sections of the distilation column.

  Parameters:
  -----------
  F : float
    Feed molar flowrate in kmol/hr (kilomoles per hour).
  xf : float
    Liquid fraction of the lower boiling boint species in the feed (unitless).
  xd : float
    Liquid fraction of the lower boiling boint species in the distilate (unitless).
  xb : float
    Liquid fraction of the lower boiling boint species in the bottoms (unitless).
  R : float (Optional)
    Reflux ratio of the rectifying section (unitless).
  q : float (Optional)
    Feed liquid fraction (unitless).

  Returns:
  -----------
  D : float
    Distilate molar flowrate in kmol/hr (kilomoles per hour).
  B : float
    Bottoms molar flowrate in kmol/hr (kilomoles per hour).
  V : float
    Vapor molar flowrate from the feed tray to the rectifying section in kmol/hr (kilomoles per hour).
  L : float
    Liquid molar flowrate from the rectifying section to the feed tray in kmol/hr (kilomoles per hour).
  Vprime : float
    Vapor molar flowrate from the stripping section to the feed tray in kmol/hr (kilomoles per hour).
  Lprime : float
    Liquid molar flowrate from the feed tray to the stripping section in kmol/hr (kilomoles per hour).
  '''
  D = F*(xf - xd)/(xd - xb)
  B = F - D
  V = None; Vprime = None
  L = None; Lprime = None
  if R != None:
    V = D * (1. + R)
    L = V - D
    if q != None:
      Vprime = V - F*(1. - q)
      Lprime = Vprime + B

  return D, B, V, L, Vprime, Lprime

# TODO #10 finish ponchon_savarit
def ponchon_savarit_graph(props: npt.ArrayLike, xf: float, yf: float, xd: float, xb: float, feedSatLiq: bool = True):
  '''
  Calculates the pochon_savarit Diagram for a bianary mixture distilation column.

  Parameters:
  -----------
  props : ArrayLike
    Chemical properties of the compounds being analyzed. Shape must be 2 x 3.
      Ex) np.array([Boiling Point Temperature (K), Average Molar Heat Capactity (kJ/mol*C), Molar Heat of Vaporization (kJ/mol) ])
  xf : float
    Liquid fraction of the lower boiling boint species in the feed (unitless).
  yf : float
    Vapor fraction of the lower boiling boint species in the feed (unitless). Corresponding y-value of xf on the equalibrium curve.
  xd : float
    Liquid fraction of the lower boiling boint species in the distilate (unitless).
  xb : float
    Liquid fraction of the lower boiling boint species in the bottoms (unitless).
  feedSatLiq : bool  # TODO #9 check if q is a valid substitute for this var
    Whether the feed is saturated liquid (vaporless / at its bubble point). False means the feed is saturated vapor (liquidless / at its dew point).

  Returns:
  -----------
  
  '''
  props = np.atleast_1d(props).reshape((-1, 3))
  if props[0, 0] > props[1, 0]:
    props = props[::-1]
  liqlineH = common.point_slope((1., 0.), (0., props[1, 1] * (props[1, 0] - props[0, 0])))
  vaplineH = common.point_slope((1., props[0, 2]), (0., props[1, 1] * (props[1, 0] - props[0, 0]) + props[1, 2]))

  if feedSatLiq:
    feedpoint = (xf, liqlineH.eval(xf))
    tiepoint = (yf, vaplineH.eval(yf))
  elif not feedSatLiq:
    feedpoint = (yf, vaplineH.eval(yf))
    tiepoint = (xf, liqlineH.eval(xf))
  else:
    return liqlineH, vaplineH #manual guess-and-check for feedpoint required

  tieline = common.point_slope(feedpoint, tiepoint)
  hd = (xd, liqlineH.eval(xd))
  hv1 = (xd, vaplineH.eval(xd))
  hdqcd = (xd, tieline.eval(xd))
  Rmin = (hdqcd - hv1) / (hv1 - hd)

  return
