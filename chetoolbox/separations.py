import numpy as np
import numpy.typing as npt
from typing import Callable
import common

def psi_solver(x: list, K: list, psi: float, tol: float = 0.01) -> common.SolutionObj[float, npt.ArrayLike, npt.ArrayLike, float, int]:
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
  return common.SolutionObj(psi = psi, x_out = x_out, y_out = y_out, error = error(psi), i = i)

def bubble_point_iterator(x:list, K:list) -> common.SolutionObj[float, npt.ArrayLike, float]:
  '''
  Intended to be used with a DePriester Chart. Calculates the vapor mole fractions & associated error, then proposes a new temperature on the DePriester chart to try.

  Parameters
  ----------
  x : list
    Component mole fractions of the liquid mixture (unitless). Must sum to 1.
  K : list
    Component equilibrium constants


  Returns
  ----------
  err : float
    associated error of the proposed bubble point temperature
  y : list
    list of vapor mole fractions based on equilibrium constants

  '''
  x = np.atleast_1d(x)
  K = np.atleast_1d(K)
  y = x * K
  err = np.sum(y) - 1 
  return common.SolutionObj(y = y, err = err)

def dew_point_iterator(y:list, K:list) -> common.SolutionObj[float, npt.ArrayLike, float]:
  '''
  Intended to be used with a DePriester Chart. Calculates the vapor mole fractions & associated error, then proposes a new temperature on the DePriester chart to try.

  Parameters
  ----------
  y : list
    Component mole fractions of the vapor mixture (unitless). Must sum to 1.
  K : list
    Component equilibrium constants


  Returns
  ----------
  err : float
    associated error of the proposed dew point temperature
  x : list
    list of liquid mole fractions based on equilibrium constants

  '''
  y = np.atleast_1d(x)
  K = np.atleast_1d(K)
  x = y / K
  err = np.sum(x) - 1 
  return common.SolutionObj(x = x, err = err)

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
  Calculates the liquid fraction of a subcooled binary liquid mixture feed.

  Parameters:
  -----------
  Cpl : float
    Specific heat of the liquid feed in J/mol*K (joules per mole Kelvin).
  heatvap : float
    Heat of vaporization of the liquid feed in J/mol (joules per mole).
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
  Calculates the liquid fraction of a superheated binary vapor mixture feed.

  Parameters:
  -----------
  Cpv : float
    Specific heat of the vapor feed in J/mol*K (joules per mole Kelvin).
  heatvap : float
    Heat of vaporization of the vapor feed in J/mol (joules per mole).
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
  Estimates an equalibrium curve for a binary mixture. Assumes constant equalibrium ratio (K1 / K2) between the two species.

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
    Equation for an equalibrium curve of a binary mixture.
  '''
  points = np.atleast_1d(points).reshape((-1, 2))
  if alpha == None:
    alpha = np.average( points[:, 1] * (1. - points[:, 0]) / (points[:, 0] * (1. - points[:, 1])) )
  return common.EqualibEq(alpha)

def mccabe_thiel_feedline(q: float, xf: float) -> common.LinearEq:
  '''
  Calculates the feed line on a McCabe Thiel Diagram for a binary mixture distilation column. Assumes equal molar heats of vaporization.

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

def mccabe_thiel_otherlines(feedline: common.LinearEq, eq_feedpoint: tuple, xd: float, xb: float, Rmin_mult: float = 1.2) -> common.SolutionObj[common.LinearEq, common.LinearEq, tuple[float, float], float]:
  '''
  Calculates the rectifying and stripping operating lines of a McCabe Thiel Diagram for a binary mixture distilation column. Assumes equal molar heats of vaporization.

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
  eq_rectifyline = common.point_conn(eq_feedpoint, (xd, xd))

  # "distilate to feedpoint" line
  Rmin = eq_rectifyline.m / (1. - eq_rectifyline.m)
  R = Rmin_mult * Rmin
  m = R / (1. + R)
  y_int = xd / (1. + R)
  rectifyline = common.LinearEq(m, y_int)

  # feedpoint
  if np.isnan(feedline.m) or np.isnan(feedline.b):
    feedpoint = (eq_feedpoint[0],rectifyline.eval(eq_feedpoint[0]))
  else:
    feedpoint = common.linear_intersect(feedline, rectifyline)

  # bottoms to feed point
  stripline = common.point_conn(feedpoint, (xb, xb))
  return common.SolutionObj(rectifyline = rectifyline, stripline = stripline, feedpoint = feedpoint, Rmin = Rmin, R = R)

def mccabe_thiel_full_est(eq_curve: common.EqualibEq, feedline: common.LinearEq, xf: float, xd: float, xb: float, Rmin_mult: float = 1.2, tol: float = .00001) -> common.SolutionObj[float, float, float, float]:
  '''
  Calculates the reflux ratio and ideal stages of a binary mixture distilation column, as well as thier ideal minimums. Uses a McCabe Thiel Diagram and assumes equal molar heats of vaporization.

  Parameters:
  -----------
  equalibrium_curve : EqualibEq
    Equation for an equalibrium curve of a binary mixture.
  feedline : LinearEq
    Feed line of a McCabe Thiel Diagram.
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
    Minimum number of ideal stages (includes reboiler and partial condenser if applicable).
  ideal_stages : float
    Number of ideal stages (includes reboiler and partial condenser if applicable).
  '''

  def err(x):
    return eq_curve.eval(x) - feedline.eval(x)
  if np.isnan(feedline.m) and np.isnan(feedline.b):
    x = xf
  else:
    x, error, i = common.iter(err, [xb, xd], tol)

  eq_feedpoint = (x, eq_curve.eval(x))
  rectifyline, stripline, feedpoint, Rmin, R = mccabe_thiel_otherlines(feedline, eq_feedpoint, xd, xb, Rmin_mult).unpack()

  y_reflect = common.LinearEq(1., 0.)
  min_stages = common.curve_bouncer(eq_curve, y_reflect, xd, xb)

  y_operlines = common.PiecewiseEq((stripline, rectifyline), (feedpoint[0],))
  ideal_stages = common.curve_bouncer(eq_curve, y_operlines, xd, xb)

  return common.SolutionObj(Rmin = Rmin, R = R, min_stages = min_stages, ideal_stages = ideal_stages)

def binary_feed_split(F: float, xf: float, xd: float, xb: float, R: float = None, q: float = None) -> tuple[float, float, float | None, float | None, float | None, float | None]:
  '''
  Calculates the distilate and bottom flow rates out of a binary mixture distilation column. Optionally calculates the internal flows between the feed tray, rectifying, and stripping sections of the distilation column.

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

def ponchon_savarit_enthalpylines(props: npt.ArrayLike) -> tuple[common.LinearEq, common.LinearEq]:
  '''
  Calculates the liquid and vapor enthalpy lines on a Pochon Savarit diagram for a binary mixture distilation column.

  Parameters:
  -----------
  props : ArrayLike
    Chemical properties of the compounds being analyzed. Shape must be 2 x 3.
      Ex) np.array([Boiling Point Temperature (K), Average Molar Heat Capactity (kJ/mol*C), Molar Heat of Vaporization (kJ/mol) ])
  
  Returns:
  -----------
  liqlineH : LinearEq
  vaplineH : LinearEq
  '''
  props = np.atleast_1d(props).reshape((-1, 3))
  if props[0, 0] > props[1, 0]:
    props = props[::-1]
  liqlineH = common.point_conn((1., 0.), (0., props[1, 1] * (props[1, 0] - props[0, 0])))
  vaplineH = common.point_conn((1., props[0, 2]), (0., props[1, 1] * (props[1, 0] - props[0, 0]) + props[1, 2]))

  return liqlineH, vaplineH

def ponchon_savarit_tieline(liqlineH: common.LinearEq, vaplineH: common.LinearEq, xf: float, yf: float, xd: float, Rmin_mult: float = 1.2) -> common.SolutionObj[common.LinearEq, float, float, float]:
  '''
  Calculates the tieline and Rmin of a Pochon Savarit diagram for a binary mixture distilation column.

  Parameters:
  -----------
  liqlineH : LinearEq

  vaplineH : LinearEq

  xf : float
    Liquid fraction of the lower boiling boint species in the feed (unitless).
  yf : float
    Vapor fraction of the lower boiling boint species in the feed (unitless). Corresponding y-value of xf on the equalibrium curve.
  xd : float
    Liquid fraction of the lower boiling boint species in the distilate (unitless).
  xb : float
    Liquid fraction of the lower boiling boint species in the bottoms (unitless).
  Rmin_mult : float
    Factor by which to excede the minimum reflux ratio, Rmin (unitless). Typical reflux ratios are between 1.05 and 1.3 times Rmin. Bounded (1, inf).

  Returns:
  -----------
  tieline : LinearEq

  Rmin : float
    Minimum reflux ratio of the rectifying section (unitless).
  R : float
    Reflux ratio of the rectifying section (unitless).
  P : float
    IDK what this represents tbh # TODO figure out what P actually is lol
  '''
  tieline = common.point_conn((xf, liqlineH.eval(xf)), (yf, vaplineH.eval(yf)))
  hd = liqlineH.eval(xd)
  hv1 = vaplineH.eval(xd)
  hdqcd = tieline.eval(xd)
  Rmin = (hdqcd - hv1) / (hv1 - hd)
  R = Rmin  * Rmin_mult
  P = R * (hv1 - hd) + hv1
  tieline = common.point_conn((xf, liqlineH.eval(xf)), (xd, P)) # tieline for real R, not Rmin

  return common.SolutionObj(tieline = tieline, Rmin = Rmin, R = R, P = P)

def ponchon_savarit_full_est(eq_curve: common.EqualibEq, liqlineH: common.LinearEq, vaplineH: common.LinearEq, Fpoint: tuple[float, float], q: bool | float, xd: float, xb: float, Rmin_mult: float, tol: float = .00001) -> tuple[float, float, float, float]:
  '''
  Calculates the liquid and vapor enthalpy lines on a Pochon Savarit diagram for a binary mixture distilation column.

  Parameters:
  -----------
  eq_curve : EqualibEq
    Equation for an equalibrium curve of a binary mixture.
  liqlineH : LinearEq

  vaplineH : LinearEq

  Fpoint : tuple[float, float] # TODO how would we be given this in point format?
    Cooridinates of the feed point on the Pochon Savarit diagram in (mol fraction (unitless), enthalpy (J/mol)) (unitless, Joules per mole).
  q : bool | float
    The liquid fraction of the incoming feed. Should be True or 1. when the feed is saturated liquid (vaporless / at its bubble point). Should be False or 0. when the feed is saturated vapor (liquidless / at its dew point).
  xd : float
    Liquid fraction of the lower boiling boint species in the distilate (unitless).
  xb : float
    Liquid fraction of the lower boiling boint species in the bottoms (unitless).
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
    Minimum number of ideal stages (includes reboiler and partial condenser if applicable).
  ideal_stages : float
    Number of ideal stages (includes reboiler and partial condenser if applicable).
  '''
  if type(q) == bool or q == 0. or q == 1.: # saturated liq / vap feed
    if q:
      xf = Fpoint[0]
      yf = eq_curve.eval(xf)
    else:
      yf = Fpoint[0]
      xf = eq_curve.inv(yf)
  else: # iteratively solve for x_f* and y_f* of the feed
    # take q as good estimate for the feedline's slope
    xf, _ = common.linear_intersect(common.point_slope(Fpoint, q), liqlineH)
    yf = eq_curve.eval(xf)
    q_alt = common.point_conn(Fpoint, (yf, vaplineH.eval(yf))).m
    tieslopes = np.array([q, q_alt])

    def error(q):
      xf, _ = common.linear_intersect(common.point_slope(Fpoint, q), liqlineH)
      return 1. - (xf + eq_curve.eval(xf))
    
    tieslope, _, _ = common.iter(error, tieslopes, tol)
    xf, _ = common.linear_intersect(common.point_slope(Fpoint, tieslope), liqlineH)
    yf = eq_curve.eval(xf)
  
  tieline, Rmin, R, P = ponchon_savarit_tieline(liqlineH, vaplineH, xf, yf, xd, Rmin_mult)

  min_stages = common.curve_bouncer(liqlineH, vaplineH, xd, xb, eq_curve.eval)

  def y_transform(y):
    line = common.point_conn( (vaplineH.inv(y), y), (xd, P))
    _, y = common.linear_intersect(line, vaplineH)
    return y
  
  ideal_stages = common.curve_bouncer(liqlineH, vaplineH, xd, xb, eq_curve.eval, y_transform)

  return Rmin, R, min_stages, ideal_stages

def multicomp_feed_split_est(feed: npt.ArrayLike, keys: tuple[int, int], spec: tuple[float, float]) -> tuple[npt.ArrayLike, npt.ArrayLike]:
  '''
  Estimates the distilate and bottoms outflow rates of a multi-component distilation column.

  Parameters:
  -----------
  feed : ArrayLike
    Molar flowrate in mol/s (moles per second) and molecular weight ii g/mol (grams per mole) of each input feed species. Shape must be N x 2.
      Ex) np.array([[448., 58.12], [36., 72.15], [23., 86.17], [39.1, 100.21], [272.2, 114.23], [31., 128.2]])
  keys : tuple[int, int]
    Indexes of the High Key Species and Low Key Species in the feed array.
  spec : tuple[float, float]
    Required molar flowrate of the High Key Species in the distilate and the Low Key Species in the bottoms in mol/s (moles per second).

  Returns:
  -----------
  distil : ArrayLike
    Molar flowrates of all species in the ditilate in mol/s (moles per second).
  bottoms : ArrayLike
    Molar flowrates of all species in the bottoms in mol/s (moles per second).
  '''
  feed = np.atleast_1d(feed).reshape((-1, 2))
  topsplit = spec[0] / feed[keys[0], 0]
  botsplit = 1. - (spec[1]) / feed[keys[1], 0]
  splitline = common.point_conn((feed[keys[1], 1], botsplit), (feed[keys[0], 1], topsplit))

  def splitest(MW: float):
    cutoff = np.max(np.c_[splitline.eval(MW)], 1, initial=0.)
    return np.min(np.c_[cutoff], 1, initial=1.)

  distil = feed[:, 0] * splitest(feed[:, 1])

  return distil, feed[:, 0] - distil

def lost_work(inlet: npt.ArrayLike, outlet: npt.ArrayLike, Q: npt.ArrayLike, T_s: npt.ArrayLike, T_0: float,  W_s: float = 0) -> float:
  '''
  Solves for the lost work of a separation process via thermodynamic analysis.  
  
  Parameters
  ----------
  inlet : ArrayLike
    The thermodynamic properties of the inlet stream in J/mol (Joules per mole), must be ordered as [n: molar flow rate of component, h: enthalpy of component, s: entropy of component] Shape must be N x 3.
      Ex) np.array([n1, h1, s1], [n2, h2, s2], [n3, h3, s3])
  outlet : ArrayLike
    The thermodynamic properties of the outlet stream in J/mol (Joules per mole), must be ordered as [n: molar flow rate of component, h: enthalpy of component, s: entropy of component] Shape must be N x 3.
      Ex) np.array([n1, h1, s1], [n2, h2, s2], [n3, h3, s3])
  Q : ArrayLike
    The reboiler and condenser heat duties in J (Joules). Shape must be 1 x 2.
      Ex) np.array([Q_reboiler, Q_condenser])
  T_s : ArrayLike
    Temperature of steam and cooling water in K (Kelvin).
      Ex) np.array([T_steam, T_CW])
  T_0 : float
    Temperature of surroundings in K (Kelvin).
  W_s : float
    Shaft work in Joules.

  Returns
  ----------
  LW : float
    Lost work in J/mol (Joules per mole).
  '''
  inlet = np.atleast_1d(inlet).reshape(-1,3)
  outlet = np.atleast_1d(outlet).reshape(-1,3)
  Q = np.atleast_1d(Q)
  T_s = np.atleast_1d(T_s)
  def b(h,s):
    return h - T_0 * s
  return np.sum(inlet[:,0] * b(inlet[:,1], inlet[:,2]) + Q[0] * (1 - T_0/T_s[0]) + W_s) - np.sum(outlet[:,0] * b(outlet[:,1], outlet[:,2]) + Q[1] * (1 - T_0/T_s[1]) + W_s)
 