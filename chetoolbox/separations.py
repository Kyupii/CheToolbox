import numpy as np
import numpy.typing as npt
from typing import Callable
from matplotlib import pyplot as plt
import common

def psi_solver(x: list, K: list, psi: float, tol: float = 0.01) -> common.SolutionObj[float, npt.NDArray, npt.NDArray, float, int]:
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
  x_out : NDArray
    Component mole fractions of the output liquid stream (unitless). 
  y_out : NDArray
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

def bubble_point_stepper(x: list, K: list) -> common.SolutionObj[float, npt.NDArray, float]:
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

def dew_point_stepper(y: list, K: list) -> common.SolutionObj[float, npt.NDArray, float]:
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

def bubble_point_antoine(x: list, ant_coeff: npt.NDArray, P: float, tol: float = .05) -> common.SolutionObj[float, npt.NDArray, npt.NDArray, npt.NDArray, float, int]:
  '''
  Iteratively solves for the bubble point temperature of a multi-component liquid mixture.

  Parameters
  ----------
  x : list
    Component mole fractions of the liquid mixture (unitless). Must sum to 1.
  ant_coeff : NDArray
    Components' coefficients for the Antoine Equation of State (unitless). Shape must be N x 3.
  P : float
    Ambient pressure of the liquid mixture in mmHg (millimeters of mercury).
  tol : float
    Largest error value to stop iterating and return.

  Returns
  ----------
  bubbleT : float
    Temperature of the liquid mixture's bubble point in C (Celcius).
  Pvap : NDArray
    Vapor pressure for each component at the bubble point temperature in mmHg (millimeters of mercury). 
  K : NDArray
    Equalibrium constant for each component at the stated pressure and bubble point temperature (units vary). 
  y : NDArray
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
    return np.sum(y, axis=0, keepdims=True) - 1.
  
  bubbleT, error, i = common.err_reduc_iterative(err, [np.max(boil_points), np.min(boil_points)], tol)
  bubbleT = bubbleT[0]; error = error[0]

  Pvap, k, y = TtoY(bubbleT)
  return common.SolutionObj(bubbleT = bubbleT, Pvap = Pvap, k = k, y = y, error = error, i = i)

def dew_point_antoine(y: list, ant_coeff: npt.NDArray, P: float, tol: float = .05) -> common.SolutionObj[float, npt.NDArray, npt.NDArray, npt.NDArray, float, int]:
  '''
  Iteratively solves for the dew point temperature of a multi-component vapor mixture.

  Parameters
  ----------
  y : list
    Component mole fractions of the vapor mixture (unitless). Must sum to 1.
  ant_coeff : NDArray
    Components' coefficients for the Antoine Equation of State (unitless). Shape must be N x 3.
  P : float
    Ambient pressure of the vapor mixture in mmHg (millimeters of mercury).
  tol : float
    Largest error value to stop iterating and return.

  Returns
  ----------
  dewT : float
    Temperature of the vapor mixture's dew point in C (Celcius).
  Pvap : NDArray
    Vapor pressure for each component at the dew point temperature in mmHg (millimeters of mercury). 
  K : NDArray
    Equalibrium constant for each component at the stated pressure and dew point temperature (units vary). 
  x : NDArray
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
    return np.sum(x, axis=0, keepdims=True) - 1.
  
  dewT, error, i = common.err_reduc_iterative(err, [np.max(boil_points), np.min(boil_points)], tol)
  dewT = dewT[0]; error = error[0]
  
  Pvap, k, y = TtoX(dewT)
  return common.SolutionObj(dewT = dewT, Pvap = Pvap, k = k, y = y, error = error, i = i)

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

def eq_curve_estim(points: npt.NDArray, alpha: float = None) -> common.EqualibEq:
  '''
  Estimates an equalibrium curve for a binary mixture. Assumes constant equalibrium ratio (K1 / K2) between the two species.

  Parameters:
  -----------
  points : NDArray
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
  if alpha is None:
    alpha = np.average( points[:, 1] * (1. - points[:, 0]) / (points[:, 0] * (1. - points[:, 1])) )
  return common.EqualibEq(alpha)

def mccabe_thiel_feedline(q: float, xf: float) -> common.LinearEq:
  '''
  Calculates the feed line on a McCabe Thiel Diagram for a binary mixture distilation column. Assumes equal molar heats of vaporization.

  Parameters:
  -----------
  q : float
    The liquid fraction of the incoming feed. Should be 1. when the feed is saturated liquid (vaporless / at its bubble point). Should be 0. when the feed is saturated vapor (liquidless / at its dew point).
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
    feedline = common.point_slope((xf, xf), m)
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
  rectifyline = common.point_slope((xd, xd), m)

  # feedpoint
  feedpoint = common.linear_intersect(feedline, rectifyline)

  # bottoms to feed point
  stripline = common.point_conn(feedpoint, (xb, xb))
  return common.SolutionObj(rectifyline = rectifyline, stripline = stripline, feedpoint = feedpoint, Rmin = Rmin, R = R)

def mccabe_thiel_full_est(eq_curve: common.EqualibEq, feedline: common.LinearEq, xf: float, xd: float, xb: float, Rmin_mult: float = 1.2, tol: float = .00001, PLOTTING_ENABLED = False) -> common.SolutionObj[float, float, float, float]:
  '''
  Calculates the reflux ratio and ideal stages of a binary mixture distilation column, as well as thier ideal minimums. Uses a McCabe Thiel diagram and assumes equal molar heats of vaporization.

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
  if np.isnan(feedline.m):
    x = xf
  else:
    x, _, _ = common.err_reduc_iterative(err, [xb, xd], tol)
    x = x[0]
  
  eq_feedpoint = (x, eq_curve.eval(x))
  rectifyline, stripline, feedpoint, Rmin, R = mccabe_thiel_otherlines(feedline, eq_feedpoint, xd, xb, Rmin_mult).unpack()

  y_reflect = common.LinearEq(1., 0.)
  min_stages = common.curve_bouncer(eq_curve, y_reflect, xd, xb)

  y_operlines = common.PiecewiseEq((stripline, rectifyline), (feedpoint[0],))

  linestograph = []
  def x_graphcapture(x):
    y = eq_curve.eval(x)
    linestograph.append(common.point_separsort((x, y), (x, y_operlines.eval(x))))
    return x
  def y_graphcapture(y):
    x = y_operlines.inv(y)
    linestograph.append(common.point_separsort((x, y), (eq_curve.inv(y), y)))
    return y
  
  ideal_stages = common.curve_bouncer(eq_curve, y_operlines, xd, xb, x_graphcapture, y_graphcapture)

  if PLOTTING_ENABLED:
    fig, ax = plt.subplots(); ax.set_title("McCabe Thiel Diagram")
    plt.xlim(0, 1); plt.ylim(0, 1)
    ax.plot([xf]*200, np.linspace(0., eq_curve.eval(xf), 200), "k")
    ax.plot([xb]*200, np.linspace(0., eq_curve.eval(xb), 200), "k")
    ax.plot([xd]*200, np.linspace(0., eq_curve.eval(xd), 200), "k")
    ax.plot(np.linspace(0., 1., 200), np.linspace(0., 1., 200), "k")
    ax.plot(np.linspace(0., 1., 200), eq_curve.eval(np.linspace(0., 1., 200)), "g")
    ax.plot(np.linspace(eq_feedpoint[0], xf, 200), feedline.eval(np.linspace(eq_feedpoint[0], xf, 200)), "m")
    ax.plot(np.linspace(xb, xd, 200), y_operlines.eval(np.linspace(xb, xd, 200)), "b")
    for dom, vals in linestograph:
      ax.plot(dom, vals, "r")
    # ax.plot(eq_feedpoint[0], eq_feedpoint[1], 'o'); ax.plot(feedpoint[0], feedpoint[1], 'o')

  return common.SolutionObj(Rmin = Rmin, R = R, min_stages = min_stages, ideal_stages = ideal_stages)

def binary_feed_split(F: float, xf: float, xd: float, xb: float, R: float = None, q: float = None) -> tuple[float, float, float | None, float | None, float | None, float | None]:
  '''
  Calculates the distilate and bottom flowrates out of a binary mixture distilation column. Optionally calculates the internal flows between the feed tray, rectifying, and stripping sections of the distilation column.

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
    The liquid fraction of the incoming feed. Should be 1. when the feed is saturated liquid (vaporless / at its bubble point). Should be 0. when the feed is saturated vapor (liquidless / at its dew point).

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

def ponchon_savarit_enthalpylines(props: npt.NDArray) -> tuple[common.LinearEq, common.LinearEq]:
  '''
  Calculates the liquid and vapor enthalpy lines on a Pochon Savarit diagram for a binary mixture distilation column.

  Parameters:
  -----------
  props : NDArray
    Chemical properties of the compounds being analyzed. Shape must be 2 x 3.
      Ex) np.array([Boiling Point Temperature (K), Average Molar Heat Capactity (kJ/mol*C), Molar Heat of Vaporization (kJ/mol) ])
  
  Returns:
  -----------
  liqlineH : LinearEq
    Enthalpy concentration line of a binary mixture in the liquid phase on a Pochon Savarit diagram.
  vaplineH : LinearEq
    Enthalpy concentration line of a binary mixture in the vapor phase on a Pochon Savarit diagram.
  '''
  props = np.atleast_1d(props).reshape((-1, 3))
  if props[0, 0] > props[1, 0]:
    props = props[::-1]
  liqlineH = common.point_conn((1., 0.), (0., props[1, 1] * (props[1, 0] - props[0, 0])))
  vaplineH = common.point_conn((1., props[0, 2]), (0., props[1, 1] * (props[1, 0] - props[0, 0]) + props[1, 2]))

  return liqlineH, vaplineH

def ponchon_savarit_tieline(liqlineH: common.LinearEq, vaplineH: common.LinearEq, xf: float, yf: float, xd: float, xb: float, Rmin_mult: float = 1.2) -> common.SolutionObj[common.LinearEq, float, float, float, float]:
  '''
  Calculates the tieline and Rmin of a Pochon Savarit diagram for a binary mixture distilation column.

  Parameters:
  -----------
  liqlineH : LinearEq
    Enthalpy concentration line of a binary mixture in the liquid phase on a Pochon Savarit diagram.
  vaplineH : LinearEq
    Enthalpy concentration line of a binary mixture in the vapor phase on a Pochon Savarit diagram.
  x_f* : float
    Liquid fraction of the lower boiling boint species in the feed (unitless). Corresponding x-value of yf on the equalibrium curve.
  y_f* : float
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
    Tie line of a Pochon Savarit diagram, which connects the feedpoint, x_f*, y_f*, P', and B' points.
  Rmin : float
    Minimum reflux ratio of the rectifying section (unitless).
  R : float
    Reflux ratio of the rectifying section (unitless).
  Hd : float
    Total enthalpy of the lower boiling point species at dew point plus the condenser's heat duty divided by the distilate flowrate 
  Hb : float
    Total enthalpy of the higher boiling point species at dew point plus the condenser's heat duty divided by the distilate flowrate 
  '''
  eq_tieline = common.point_conn((xf, liqlineH.eval(xf)), (yf, vaplineH.eval(yf))) # tieline for Rmin
  hd = liqlineH.eval(xd)
  hv1 = vaplineH.eval(xd)
  eq_hdqcd = eq_tieline.eval(xd)
  Rmin = (eq_hdqcd - hv1) / (hv1 - hd)
  R = Rmin  * Rmin_mult
  Hd = R * (hv1 - hd) + hv1
  tieline = common.point_conn((xf, liqlineH.eval(xf)), (xd, Hd)) # tieline for real R, not Rmin
  Hb = tieline.eval(xb)

  return common.SolutionObj(tieline = tieline, Rmin = Rmin, R = R, Hd = Hd, Hb = Hb)

def ponchon_savarit_full_est(eq_curve: common.EqualibEq, liqlineH: common.LinearEq, vaplineH: common.LinearEq, Fpoint: tuple[float, float], q: bool | float, xd: float, xb: float, Rmin_mult: float, tol: float = .00001, PLOTTING_ENABLED = False) -> common.SolutionObj[common.LinearEq, float, float, float, float]:
  '''
  Calculates the liquid and vapor enthalpy lines on a Pochon Savarit diagram for a binary mixture distilation column.

  Parameters:
  -----------
  eq_curve : EqualibEq
    Equation for an equalibrium curve of a binary mixture.
  liqlineH : LinearEq
    Enthalpy concentration line of a binary mixture in the liquid phase on a Pochon Savarit diagram.
  vaplineH : LinearEq
    Enthalpy concentration line of a binary mixture in the vapor phase on a Pochon Savarit diagram.
  Fpoint : tuple[float, float]
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
    
    tieslope, _, _ = common.err_reduc_iterative(error, tieslopes, tol)
    xf, _ = common.linear_intersect(common.point_slope(Fpoint, tieslope), liqlineH)
    xf = xf[0]
    yf = eq_curve.eval(xf)
  
  tieline, Rmin, R, Hp, Hb = ponchon_savarit_tieline(liqlineH, vaplineH, xf, yf, xd, xb, Rmin_mult).unpack()

  def y_transform(y):
    return vaplineH.eval(liqlineH.inv(y))

  min_stages = common.curve_bouncer(vaplineH, liqlineH, xd, xb, eq_curve.inv, y_transform)
  
  global linestograph
  linestograph = []
  def y_transform(y):
    x = liqlineH.inv(y)
    if x > Fpoint[0]:
      connectpoint = (xd, Hp)
    else:
      connectpoint = (xb, Hb)
    line = common.point_conn( (x, y), connectpoint)
    xnext, ynext = common.linear_intersect(line, vaplineH)
    if PLOTTING_ENABLED:
      if x > Fpoint[0]:
        plot1 = common.point_separsort((x, line.eval(x)), connectpoint)
      else:
        plot1 = common.point_separsort((xnext, line.eval(xnext)), connectpoint)
      plot2 = common.point_separsort((x, y), (eq_curve.eval(x), vaplineH.eval(eq_curve.eval(x))))
      linestograph.extend([plot1, plot2])
    return ynext
  
  ideal_stages = common.curve_bouncer(vaplineH, liqlineH, xd, xb, eq_curve.inv, y_transform)

  if PLOTTING_ENABLED:
    fig, ax = plt.subplots(); ax.set_title("Pochon Savarit Diagram")
    plt.xlim(0, 1); plt.ylim(Hb * 1.1, Hp * 1.1)
    x = np.linspace(0., 1., 200)
    ax.plot([xf]*200, np.linspace(Hb * 1.1, Hp * 1.1, 200), "k")
    ax.plot([xb]*200, np.linspace(Hb * 1.1, Hp * 1.1, 200), "k")
    ax.plot([xd]*200, np.linspace(Hb * 1.1, Hp * 1.1, 200), "k")
    ax.plot(x, liqlineH.eval(x), "g")
    ax.plot(x, vaplineH.eval(x), "g")
    ax.plot(np.linspace(xb, xd, 200), tieline.eval(np.linspace(xb, xd, 200)), "y")
    for i, domsvals in enumerate(linestograph):
      ax.plot(*domsvals, "rb"[i%2])
    
  return common.SolutionObj(tieline = tieline, Rmin = Rmin, R = R, min_stages = min_stages, ideal_stages = ideal_stages)

def lost_work(inlet: npt.NDArray, outlet: npt.NDArray, Q: npt.NDArray, T_s: npt.NDArray, T_0: float,  W_s: float = 0) -> float:
  '''
  Solves for the lost work of a separation process via thermodynamic analysis.  
  
  Parameters
  ----------
  inlet : NDArray
    The thermodynamic properties of the inlet stream in J/mol (Joules per mole), must be ordered as [n: molar flowrate of component, h: enthalpy of component, s: entropy of component] Shape must be N x 3.
      Ex) np.array([n1, h1, s1], [n2, h2, s2], [n3, h3, s3])
  outlet : NDArray
    The thermodynamic properties of the outlet stream in J/mol (Joules per mole), must be ordered as [n: molar flowrate of component, h: enthalpy of component, s: entropy of component] Shape must be N x 3.
      Ex) np.array([n1, h1, s1], [n2, h2, s2], [n3, h3, s3])
  Q : NDArray
    The reboiler and condenser heat duties in J (Joules). Shape must be 1 x 2.
      Ex) np.array([Q_reboiler, Q_condenser])
  T_s : NDArray
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

def multicomp_feed_split_est(F_i: npt.NDArray, MW: npt.NDArray, keys: tuple[int, int], spec: tuple[float, float]) -> tuple[npt.NDArray, npt.NDArray]:
  '''
  Estimates the distilate and bottoms outflowrates of a multi-component distilation column.
  
  Parameters:
  -----------
  F_i : NDArray
    Molar flowrate of each input feed species. Shape must be N.
      Ex) np.array([[448., 58.12], [36., 72.15], [23., 86.17], [39.1, 100.21], [272.2, 114.23], [31., 128.2]])
  MW : NDArray
    Molecular weight in g/mol (grams per mole) of each input feed species. Length must be N.
      Ex) np.array([[448., 58.12], [36., 72.15], [23., 86.17], [39.1, 100.21], [272.2, 114.23], [31., 128.2]])
  keys : tuple[int, int]
    Indexes of the High Key Species and Low Key Species in the feed array.
  spec : tuple[float, float]
    Required molar flowrate of the High Key Species in the distilate and the Low Key Species in the bottoms in mol/s (moles per second).
  
  Returns:
  -----------
  distil : NDArray
    Molar flowrates of all species in the ditilate in mol/s (moles per second).
  bottoms : NDArray
    Molar flowrates of all species in the bottoms in mol/s (moles per second).
  '''
  F_i = np.atleast_1d(F_i).reshape((-1, 2))
  topsplit = spec[0] / F_i[keys[0]]
  botsplit = 1. - (spec[1]) / F_i[keys[1]]
  splitline = common.point_conn((MW[keys[1]], botsplit), (MW[keys[0]], topsplit))
  
  def splitest(MW: float):
    cutoff = np.max(np.c_[splitline.eval(MW)], 1, initial=0.)
    return np.min(np.c_[cutoff], 1, initial=1.)
  
  distil = F_i * splitest(MW)
  
  return distil, F_i - distil

def fenske_plates(a_lk_hk_DB: npt.NDArray, x_lk_DB: npt.NDArray, x_hk_DB: npt.NDArray) -> float:
  '''
  Calculates the minimum number of stages for a multi-component distillation tower using the Fenske equation. Deviation of the relative volatilities of the light key compound and heavy key compound across the column from the geometric mean must be less than 20%.
  
  Parameters:
  -----------
  a_lk_hk_DB : NDArray
    Relative volatility of the light key compound to the heavy key compound at the final distilate plate and final reboiler plate. a_lk_hk = K_lk / K_hk.
      Ex) np.array([a_lk_hk_D, a_lk_hk_B])
  x_lk_DB : NDArray
    Liquid mole fractions of the light key component in the distillate and bottom streams. 
      Ex) np.array([x_lk_D, x_lk_B])
  x_hk_DB : NDArray
    Liquid mole fractions of the heavy key component in the distillate and bottom streams. 
      Ex) np.array([x_hk_D, x_hk_B])

  Returns
  ----------
  N_min : float
    Minimum number of stages for a multi-component distillation tower.
  '''
  a_lk_hk_DB = np.atleast_1d(a_lk_hk_DB)
  x_lk_DB = np.atleast_1d(x_lk_DB); x_hk_DB = np.atleast_1d(x_hk_DB)
  alpha_m = np.sqrt(a_lk_hk_DB[0] * a_lk_hk_DB[1])
  if np.abs((a_lk_hk_DB[0] - a_lk_hk_DB[1]) / alpha_m) > 0.20:
    raise Exception('Fenske is not valid. Use Winn equation') 
  else:
    return np.log10((x_lk_DB[0] / x_lk_DB[1]) * (x_hk_DB[1] / x_hk_DB[0])) / np.log10(alpha_m)

def fenske_feed_split(F_i: npt.NDArray, a_i_hk: npt.NDArray, N_min: float, D_hk: float, B_hk: float) -> common.SolutionObj[npt.NDArray, npt.NDArray]:
  '''
  Calculates the molar flowrates of non-key components in the distillate and bottoms streams of a multi-component distillation using the Fenske equations. Deviation of the relative volatilities of the light key compound and heavy key compound across the column must be less than 20%.
  
  Parameters:
  -----------
  F_i : NDArray
    Molar flowrates of all non-key components in the feed stream.
  a_i_hk : NDArray
    Relative volatilities of all non-key components (relative to the heavy key).
  N_min : float
    Minimum number of stages for a multi-component distillation tower.
  D_hk : float
    Molar flowrate of heavy key component in distillate stream.
  B_hk : float
    Molar flowrate of heavy key component in bottom stream.
  Returns
  ----------
  D_i : NDArray
    Molar flowrates of all non-key components in the distillate stream.
  B_i : NDArray
    Molar flowrates of all non-key components in the bottoms stream.
  '''
  F_i = np.atleast_1d(F_i); a_i_hk = np.atleast_1d(a_i_hk)
  denom = (D_hk / B_hk) * a_i_hk**N_min
  B_i = F_i / (1. + denom)
  D_i = F_i * denom / (1. + denom)
  tops = D_i > B_i
  D_i[tops] = F_i[tops] - B_i[tops]
  B_i[~tops] = F_i[~tops] - D_i[~tops]
  return common.SolutionObj(D_i = D_i, B_i = B_i)

def winn_coeff_est(K_i: npt.NDArray, K_hk: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
  '''
  Estimates the zeta coefficient and iota power of binary equalibrium pairs for use in the Winn equation.
  
  Parameters:
  -----------
  K_i : NDArray
    Equilibrium constants of non-key components at two or more points in the distillation column. Shape must be N x M, M >= 2.
      ex) np.array([[K_1_F, K_1_D, K_1_B], [K_2_F, K_2_D, K_2_B], [K_3_F, K_3_D, K_3_B]])
  K_hk : NDArray
    Equilibrium constants of the heavy key component at two or more points in the distillation column. Length must be M >= 2.
      ex) np.array([K_hk_F, K_hk_D, K_hk_B])

  Returns
  ----------
  phi : npt.NDArray
    Exponent of the Winn K-transform function.
  logzeta : npt.NDArray
    Log10 of the coefficient of the Winn K-transform function.
  '''
  K_i = np.log10(np.atleast_2d(K_i).reshape(-1, 2))
  K_hk = np.log10(np.atleast_1d(K_hk))
  points = np.vstack((K_i.flatten("A"), np.repeat(np.c_[K_hk], len(K_i[:, 0]), axis=1).flatten("F"))).T.reshape(-1, len(K_hk), 2)
  # ^^^ acrane array wizardry
  lines = [common.point_conn(v[:-1, :], v[1:, :], avgmode=True) for v in points]
  phi = np.array([line.m for line in lines]).astype(float)
  logzeta = np.array([line.b for line in lines]).astype(float)
  return phi, logzeta

def winn_plates(K_lk: npt.NDArray, K_hk: npt.NDArray, x_lk_DB: npt.NDArray, x_hk_DB: npt.NDArray) -> float:
  '''
  Calculates the minimum number of stages for a multi-component distillation tower using the Winn equation, alongside a graphical transformation.
  
  Parameters:
  -----------
  K_lk : NDArray
    Equilibrium constants of the light key component at two or more points in the distillation column. Length must be M >= 2.
      ex) np.array([K_lk_F, K_lk_D, K_lk_B])
  K_hk : NDArray
    Equilibrium constants of the heavy key component at two or more points in the distillation column. Length must be M >= 2.
      ex) np.array([K_hk_F, K_hk_D, K_hk_B])
  x_lk : NDArray
    Liquid mole fractions of the light key component in the distillate and bottom streams. 
      Ex) np.array([x_lk_D, x_lk_B])
  x_hk : NDArray
    Liquid mole fractions of the heavy key component in the distillate and bottom streams. 
      Ex) np.array([x_hk_D, x_hk_B])

  Returns
  ----------
  N_min : float
    Minimum number of stages of a multi-component distillation tower
  '''
  K_lk = np.atleast_1d(K_lk); K_hk = np.atleast_1d(K_hk)
  x_lk_DB = np.atleast_1d(x_lk_DB); x_hk_DB = np.atleast_1d(x_hk_DB)
  phi, logzeta = winn_coeff_est(K_lk, K_hk)
  N_min = np.log10((x_lk_DB[0] / x_lk_DB[1]) * (x_hk_DB[1] / x_hk_DB[0]))**phi / logzeta 
  return N_min

def winn_feed_split(F_i: npt.NDArray, K_i: npt.NDArray, K_hk: npt.NDArray, N_min: float, D: float, B: float, D_hk: float, B_hk: float) -> common.SolutionObj[npt.NDArray, npt.NDArray]:
  '''
  Calculates the molar flowrates of non-key components in the distillate and bottoms streams of a multi-component distillation using the Winn equations. 
  
  Parameters:
  -----------
  F_i : NDArray
    Molar flowrates of all non-key components in the feed stream.
      Ex) np.array([F_1, F_2, F_3])
  K_i : NDArray
    Equilibrium constants of non-key components at two or more points in the distillation column. Shape must be N x M, M >= 2.
      Ex) np.array([[K_1_F, K_1_D, K_1_B], [K_2_F, K_2_D, K_2_B], [K_3_F, K_3_D, K_3_B]])
  K_hk : NDArray
    Equilibrium constants of the heavy key component at two or more points in the distillation column. Length must be M >= 2.
      Ex) np.array([K_hk_F, K_hk_D, K_hk_B])
  N_min : float
    Minimum number of stages for a multi-component distillation tower.
  D : float
    Molar flowrate of the distillate stream.
  B : float
    Molar flowrate of the bottoms stream.
  D_hk : float
    Molar flowrate of heavy key component in distillate stream.
  B_hk : float
    Molar flowrate of heavy key component in bottoms stream.
  Returns
  ----------
  D_i : NDArray
    Molar flowrates of all non-key components in the distillate stream.
  B_i : NDArray
    Molar flowrates of all non-key components in the bottoms stream.
  '''
  F_i = np.atleast_1d(F_i)
  K_i = np.atleast_2d(K_i).reshape(-1, 2); K_hk = np.atleast_1d(K_hk)
  phi, logzeta = winn_coeff_est(K_i, K_hk)
  denom = (10.**logzeta)**N_min / (B_hk / D_hk)**phi * (B / D)**(1. - phi)
  print(denom)
  B_i = F_i / (1. + denom)
  D_i = F_i / (1. + 1. / denom)
  tops = D_i > B_i
  D_i[tops] = F_i[tops] - B_i[tops]
  B_i[~tops] = F_i[~tops] - D_i[~tops]
  common.SolutionObj(D_i = D_i, B_i = B_i)

def underwood_type1(x_i_F: npt.NDArray, a_i_hk_F: npt.NDArray, x_lk_FD: npt.NDArray, x_hk_FD: npt.NDArray, a_lk_hk_F: float, F: float, F_liq: float, D: float) -> common.SolutionObj[npt.NDArray, npt.NDArray, float, bool]:
  '''
  Calculates the minimum reflux ratio and accompanying outflow streams of a Type I System (full component distribution) using the Underwood equations. Can detect if a distilation column fails to meet Type I requirements and is actually Type II.
  
  Parameters:
  -----------
  x_i_F : NDArray
    Liquid mole fractions of all non-key components in the feed stream.
  a_i_hk_F : NDArray
    Relative volatilities of all non-key components relative to the heavy key at the feed plate.
  x_lk_FD : NDArray
    Liquid mole fractions of the light key component in the feed and distillate streams. 
      Ex) np.array([x_lk_F, x_lk_D])
  x_hk_FD : NDArray
    Liquid mole fractions of the heavy key component in the feed and distillate streams. 
      Ex) np.array([x_hk_F, x_hk_D])
  a_lk_hk_F : float
    Relative volatility of the light key compound to the heavy key compound at the feed plate.
  F : float
    Molar flowrate of the feed stream.
  F_liq : float
    Molar flowrate of the liquid portion of feed stream.
  D : float
    Molar flowrate of the distilate stream.
  
  Returns
  ----------
  D_i_Rmin : NDArray
    Molar flowrate of all non-key components in the distilate stream at minimum reflux.
  B_i_Rmin : NDArray
    Molar flowrate of all non-key components in the bottoms stream at minimum reflux.
  R_min : float
    Minimum reflux ratio of the a distilation column as a Type I System.
  typeII : NDArray
    If a component failed to distribute, meaning the distilation column is Type II.
  '''
  x_i_F = np.atleast_1d(x_i_F)
  a_i_hk_F = np.atleast_1d(a_i_hk_F)
  x_lk_FD = np.atleast_1d(x_lk_FD); x_hk_FD = np.atleast_1d(x_hk_FD)
  heavy = D * x_lk_FD[1] / (F_liq * x_lk_FD[0]) - a_lk_hk_F * (D * x_hk_FD[1] / (F_liq * x_hk_FD[0]))
  L_inf = heavy * F_liq / (a_lk_hk_F - 1.)
  R_min = L_inf / D
  
  # distribution at R_min
  lkgalf = D * x_lk_FD[1] * (a_i_hk_F - 1.) / ((a_lk_hk_F - 1.) * F_liq * x_lk_FD[0])
  hkhlaf = (a_lk_hk_F - a_i_hk_F) * D * x_hk_FD[1] / ((a_lk_hk_F - 1.) * F_liq * x_hk_FD[0])
  D_i_Rmin = (lkgalf + hkhlaf) * F_liq * x_i_F # TODO check if this is actually what you are supposed to do here
  B_i_Rmin = F * x_i_F - D_i_Rmin 
  
  cond = (lkgalf + hkhlaf) * F_liq / F # TODO check if this is actually what you are supposed to do here
  typeII = np.any((cond < 0, cond > 1) , axis=1)
  return common.SolutionObj(D_i_Rmin = D_i_Rmin, B_i_Rmin = B_i_Rmin, R_min = R_min, typeII = typeII)

def underwood_type2(x_i_F: npt.NDArray, a_i_hk_F: npt.NDArray, typeII: npt.NDArray, psi: float):
  '''
  Calculates Type II.
  
  Parameters:
  -----------
  x_i_F : NDArray
    Liquid mole fractions of all components in the feed stream.
  a_i_hk_F : NDArray
    Relative volatilities of all components relative to the heavy key at the feed plate.
  typeII : NDArray
    If a component failed to distribute, meaning the distilation column is Type II.
  '''
  x_i_F = np.atleast_1d(x_i_F)
  a_i_hk_F = np.atleast_1d(a_i_hk_F)
  typeII = np.atleast_1d(typeII)
  tIa = a_i_hk_F[~typeII]
  thetaranges = np.linspace(tIa[:-1], tIa[1:], 10).flatten("F")
  thetasets = np.vstack(np.lib.stride_tricks.sliding_window_view(thetaranges, 2))
  
  def err(theta):
    return psi - np.sum(a_i_hk_F * x_i_F / (a_i_hk_F - theta), keepdims=True)
  
  theta, _, _ = common.err_reduc_iterative(err, thetasets)
  
  theta.sort(); theta = theta[theta > np.min(tIa)]; theta = theta[theta < np.max(tIa)]
  ltnind = (np.c_[theta] > tIa).sum(axis=1)
  split_ind = np.where(ltnind[:-1] != ltnind[1:])[0] + 1
  thetasgrouped = np.split(theta, split_ind) # theoretically will be len(tIa) - 1 groups of thetas that all converged to approx. the same number
  
  # no idea how to use theta to solve for component distilate flowrates when there can be arbitrarily many unknowns!!
  
  return

def gilliland(Nmin: float, Rmin: float, Rmin_mult: float = 1.3) -> float:
  '''
  Solves for the number of real trays required to operate a distillation column  
  
  Parameters
  ----------
  Nmin : float
    Minimum number of trays required to operate a column. Usually calculated from the Fenske equation 
  Rmin : float
    Minimum reflux ratio required to operate a column.
  Rmin_mult : float
    Multiplcation factor for which the actual reflux ratio is found. Default is taken to be 1.3
  
  Returns
  ----------
  N : float
    Actual number of trays required to operate the column
  '''
  R = Rmin_mult * Rmin
  X = (R - Rmin) / (R + 1)
  Y = 1 - np.exp((1 + 54.4 * X) / (11 + 117.2 * X) * ((X - 1) / np.sqrt(X)))
  if Rmin > 0.53 or Rmin < 0.53 or Nmin <3.4 or Nmin > 60.3:
    raise Exception('Gilliland correlation is not valid in this case!')
  return (Nmin + Y) / (1 - Y)
