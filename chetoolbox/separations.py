import numpy as np
import numpy.typing as npt
from typing import Callable
from matplotlib import pyplot as plt
import common, props

def psi_solver(x: list, K: list, psi: float, tol: float = 0.01) -> common.SolutionObj[float, npt.NDArray, npt.NDArray, float, int]:
  '''
  Iteratively solves for the vapor/liquid output feed ratio psi (Ψ) of a multi-component fluid stream.  
  
  Parameters
  ----------
  x : list
    Component mole fractions of the liquid input feed (unitless). Must sum to 1.
  K : list
    Equilibrium constant for each component at specific temperature and pressure (units vary). Length must equal x.
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

def bubble_press_antoine(x: list, ant_coeff: npt.NDArray, T: float | npt.NDArray) -> common.SolutionObj[float | npt.NDArray, npt.NDArray]:
  '''
  Calculates the bubble point pressure of a multi-component liquid mixture.
  
  Parameters
  ----------
  x : list
    Component mole fractions of the liquid mixture (unitless). Must sum to 1.
  ant_coeff : NDArray
    Coefficients for the Antoine Equation of State for all components (unitless). Shape must be N x 3.
  T : float
    Ambient temperature of the liquid mixture in K (Kelvin).
  
  Returns
  ----------
  bubbleP : float
    Pressure of the liquid mixture's bubble point in mmHg (millimeters of mercury).
  Pvaps : NDArray
    Vapor pressures of the liquid mixture's components in mmHg (millimeters of mercury).
  '''
  # resources for me, the coder, a moron: https://en.wikibooks.org/wiki/Introduction_to_Chemical_Engineering_Processes
  x = np.atleast_1d(x)
  ant_coeff = np.atleast_1d(ant_coeff).reshape(-1, 3)
  Pvaps = common.antoine_P(ant_coeff, T)
  bubbleP = np.sum(Pvaps*x, axis=1)
  return common.SolutionObj(bubbleP=bubbleP, Pvaps=Pvaps)

def dew_press_antoine(y: list, ant_coeff: npt.NDArray, T: float | npt.NDArray) -> common.SolutionObj[float | npt.NDArray, npt.NDArray]:
  '''
  Calculates the dew point pressure of a multi-component vapor mixture.
  
  Parameters
  ----------
  y : list
    Component mole fractions of the vapor mixture (unitless). Must sum to 1.
  ant_coeff : NDArray
    Coefficients for the Antoine Equation of State for all components (unitless). Shape must be N x 3.
  T : float
    Ambient temperature of the liquid mixture in K (Kelvin).
  
  Returns
  ----------
  dewP : float
    Pressure of the the vapor mixture's dew point in mmHg (millimeters of mercury).
  Pvaps : NDArray
    Vapor pressures of the vapor mixture's components in mmHg (millimeters of mercury).
  '''
  y = np.atleast_1d(y)
  ant_coeff = np.atleast_1d(ant_coeff).reshape(-1, 3)
  Pvaps = common.antoine_P(ant_coeff, T)
  dewP = 1. / np.sum(y / Pvaps, axis=1)
  return common.SolutionObj(dewP=dewP, Pvaps=Pvaps)

def bubble_temp_antoine(x: list, ant_coeff: npt.NDArray, P: float) -> common.SolutionObj[float | npt.NDArray, npt.NDArray]:
  '''
  Iteratively solves for the bubble point temperature of a multi-component liquid mixture.
  
  Parameters
  ----------
  x : list
    Component mole fractions of the liquid mixture (unitless). Must sum to 1.
  ant_coeff : NDArray
    Coefficients for the Antoine Equation of State for all components (unitless). Shape must be N x 3.
  P : float
    Ambient pressure of the liquid mixture in mmHg (millimeters of mercury).
  
  Returns
  ----------
  bubbleT : float
    Temperature of the liquid mixture's bubble point in K (Kelvin).
  '''
  x = np.atleast_1d(x); P = np.c_[np.atleast_1d(P)]
  ant_coeff = np.atleast_1d(ant_coeff).reshape(-1, 3)
  Tvaps = common.antoine_T(ant_coeff, P)
  
  extrema = np.hstack([Tvaps.min(axis=1, keepdims=True), Tvaps.max(axis=1, keepdims=True)])
  match = extrema[:, 0] == extrema[:, 1]
  if match.any():
    extrema[match] = extrema[match] * np.array([.5, 1.5])
  
  def err(T):
    return bubble_press_antoine(x, ant_coeff, T.flatten("A")).bubbleP.reshape(-1, 2) - P
  
  T, _, _ = common.err_reduc_iterative(err, extrema)
  return T

def dew_temp_antoine(y: list, ant_coeff: npt.NDArray, P: float) -> common.SolutionObj[float | npt.NDArray, npt.NDArray]:
  '''
  Iteratively solves for the dew point temperature of a multi-component liquid mixture.
  
  Parameters
  ----------
  x : list
    Component mole fractions of the liquid mixture (unitless). Must sum to 1.
  ant_coeff : NDArray
    Coefficients for the Antoine Equation of State for all components (unitless). Shape must be N x 3.
  P : float
    Ambient pressure of the liquid mixture in mmHg (millimeters of mercury).
  
  Returns
  ----------
  dewT : float
    Temperature of the liquid mixture's dew point in K (Kelvin).
  '''
  y = np.atleast_1d(y); P = np.c_[np.atleast_1d(P)]
  ant_coeff = np.atleast_1d(ant_coeff).reshape(-1, 3)
  Tvaps = common.antoine_T(ant_coeff, P)
  
  extrema = np.hstack([Tvaps.min(axis=1, keepdims=True), Tvaps.max(axis=1, keepdims=True)])
  match = extrema[:, 0] == extrema[:, 1]
  if match.any():
    extrema[match] = extrema[match] * np.array([.5, 1.5])
  
  def err(T):
    return dew_press_antoine(y, ant_coeff, T.flatten("A")).dewP.reshape(-1, 2) - P
  
  T, _, _ = common.err_reduc_iterative(err, extrema)
  return T


# TODO finish these stepper functions, or delete them
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
    Molar flowrate of each input feed species. Length must be N.
      Ex) np.array([448., 36., 23., 39.1, 272.2, 31.])
  MW : NDArray
    Molecular weight in g/mol (grams per mole) of each input feed species. Length must be N.
      Ex) np.array([58.12, 72.15, 86.17, 100.21, 114.23, 128.2])
  keys : tuple[int, int]
    Indexes of the Light Key species and Heavy Key species in the feed array.
  spec : tuple[float, float]
    Required molar flowrate of the Light Key species in the distilate and Heavy Key species in the bottoms.
  
  Returns:
  -----------
  D_i : NDArray
    Molar flowrates of all components in the distilate stream.
  B_i : NDArray
    Molar flowrates of all components in the bottoms stream.
  '''
  F_i = np.atleast_1d(F_i); MW = np.atleast_1d(MW)
  topsplit = spec[0] / F_i[keys[0]]
  botsplit = 1. - (spec[1]) / F_i[keys[1]]
  splitline = common.point_conn((MW[keys[1]], botsplit), (MW[keys[0]], topsplit))
  
  def splitest(MW: float):
    cutoff = np.max(np.c_[splitline.eval(MW)], 1, initial=0.)
    return np.min(np.c_[cutoff], 1, initial=1.)
  
  D_i = F_i * splitest(MW)
  B_i = F_i - D_i
  return D_i, B_i

def multicomp_column_cond(ant_coeff: npt.NDArray, D_i: npt.NDArray, B_i: npt.NDArray, T_D: float = 312.15, T_decomp: float | None = None, numplates: float | None = None, vacuumColumn: bool = False, decompSafeFac: float = .5) -> tuple[npt.NDArray, str]:
  '''
  Calcualtes the pressure across a distilation column.
  
  Parameters:
  -----------
  ant_coeff : NDArray
    Components' coefficients for the Antoine Equation of State (unitless). Shape must be N x 3.
  D_i : NDArray
    Molar flowrates of all components in the distilate stream.
  B_i : NDArray
    Molar flowrates of all components in the bottoms stream.
  T_D : float
    Temperature of the distillate liquid in the reflux drum in K (Kelvin). Assumes 49 C (Celcius) == 312.15 Kelvin (K) by default.
  T_decomp : float
    Temperature of decomposition of the bottoms product in K (Kelvin).
  numplates : float
    Number of plates in the distilation column, if known.
  vacuumColumn : bool
    If the distilation column is operating below ambient pressure.
  decompSafeFac : float
    Ratio of the maximum reboiler temperature to the bottom product's decompostion temperature.
  
  Returns:
  -----------
  T_and_P : NDArray
    Temperature in K (Kelvin) and pressure in psia (absolute pounds per square inch) pairs for the top, average, and bottom of the distilation column.
  condenserType : str
    Type of condenser that ought to be used at the calculated distilate pressure.
  '''
  D_i = np.atleast_1d(D_i); B_i = np.atleast_1d(B_i)
  ant_coeff = np.atleast_1d(ant_coeff).reshape(-1, 3)
  x_D = D_i / D_i.sum(); x_B = B_i / B_i.sum()
  P_reflux = common.UnitConv.press(bubble_press_antoine(x_D, ant_coeff, T_D), "mmHg", "psia")
  P_reflux = np.maximum(P_reflux, 30.)
  condenserType = "Total Condenser"
  if P_reflux >= 215.:
    P_reflux = common.UnitConv.press(dew_press_antoine(x_D, ant_coeff, T_D), "mmHg", "psia")
    condenserType = "Partial Condenser"
  if P_reflux > 365.:
    P_reflux = np.minimum(P_reflux, 415.)
    condenserType = "Partial Condenser with Refridgerant"
  
  P_top = P_reflux + 5.
  # TODO how to calculate temperature at the top of the column!? dew or bubble?
  # T_top = dew_temp_antoine(x_D, ant_coeff, common.UnitConv.press(P_top, "psia", "mmHg"))
  T_top = bubble_temp_antoine(x_D, ant_coeff, common.UnitConv.press(P_top, "psia", "mmHg"))
  if numplates is None:
    dP = 5.
  else:
    dP = (.05 if vacuumColumn else .1) * numplates
  P_bot = P_top + dP
  T_bot = bubble_temp_antoine(x_B, ant_coeff, common.UnitConv.press(P_bot, "psia", "mmHg"))
  T_and_P = np.array([[T_top, P_top], [np.average([T_top, T_bot]), np.average([P_top, P_bot])], [T_bot, P_bot]])
  
  if T_decomp is not None and T_bot > decompSafeFac * T_decomp:
    P_reflux = bubble_press_antoine(x_B, decompSafeFac * T_decomp) - common.UnitConv.press(dP + 5., "psia", "mmHg")
    T_D = dew_temp_antoine(x_D, ant_coeff, P_reflux) if P_reflux >= 215. else bubble_temp_antoine(x_D, ant_coeff, P_reflux)
    T_and_P, condenserType = multicomp_column_cond(ant_coeff, D_i, B_i, T_D, T_decomp, numplates, vacuumColumn, decompSafeFac)
  
  return T_and_P, condenserType

def fenske_plates(a_i_hk: npt.NDArray, D_i: npt.NDArray, B_i: npt.NDArray, keys: tuple[int, int]) -> float:
  '''
  Calculates the minimum number of stages for a multi-component distillation tower using the Fenske equation. Deviation of the relative volatilities of the light key compound and heavy key compound across the column from the geometric mean must be less than 20%.
  
  Parameters:
  -----------
  a_i_hk : NDArray
    Relative volatility of each compound to the heavy key compound at the final distilate plate, average column conditions, and final reboiler plate (unitless).
  D_i : NDArray
    Molar flowrates of all components in the distilate stream.
  B_i : NDArray
    Molar flowrates of all components in the bottoms stream.
  keys : tuple[int, int]
    Indexes of the Light Key species and Heavy Key species in the feed array.
  
  Returns:
  ----------
  N_min : float
    Minimum number of stages for a multi-component distillation tower.
  '''
  a_i_hk = np.atleast_2d(a_i_hk)
  D_i = np.atleast_1d(D_i); B_i = np.atleast_1d(B_i)
  x_D = D_i / D_i.sum(); x_B = B_i / B_i.sum()
  alpha_m = np.sqrt(a_i_hk[0, keys[0]] * a_i_hk[2, keys[0]])
  if np.abs((a_i_hk[0, keys[0]] - a_i_hk[2, keys[0]]) / alpha_m) > 0.20:
    raise Exception('Fenske is not valid. Use Winn equation') 
  else:
    return np.log10((x_D[keys[0]] / x_B[keys[0]]) * (x_B[keys[1]] / x_D[keys[1]])) / np.log10(alpha_m)

def fenske_feed_split(a_i_hk: npt.NDArray, F_i: npt.NDArray, D_i: npt.NDArray, B_i: npt.NDArray, keys: tuple[int, int], spec: tuple[float, float]) -> common.SolutionObj[npt.NDArray, npt.NDArray, float]:
  '''
  Calculates the molar flowrates of non-key components in the distillate and bottoms streams of a multi-component distillation using the Fenske equations. Deviation of the relative volatilities of the light key compound and heavy key compound across the column must be less than 20%.
  
  Parameters:
  -----------
  a_i_hk : NDArray
    Relative volatility of each compound to the heavy key compound at the final distilate plate, average column conditions, and final reboiler plate (unitless).
  F_i : NDArray
    Molar flowrates of all components in the feed stream.
  D_i : NDArray
    Molar flowrates of all components in the distilate stream.
  B_i : NDArray
    Molar flowrates of all components in the bottoms stream. 
  keys : tuple[int, int]
    Indexes of the Light Key species and Heavy Key species in the feed array.
  spec : tuple[float, float]
    Required molar flowrate of the Light Key species in the distilate and Heavy Key species in the bottoms.
  
  
  Returns:
  ----------
  D_i : NDArray
    Improved molar flowrates of all components in the distilate stream.
  B_i : NDArray
    Improved molar flowrates of all components in the bottoms stream. 
  N_min : float
    Minimum number of stages for a multi-component distillation tower.
  '''
  a_i_hk = np.atleast_2d(a_i_hk)
  F_i = np.atleast_1d(F_i); D_i = np.atleast_1d(D_i); B_i = np.atleast_1d(B_i)
  N_min = fenske_plates(a_i_hk, D_i, B_i, keys)
  a_i_hk_m = np.sqrt(a_i_hk[0] * a_i_hk[2])
  denom = a_i_hk_m**N_min * (F_i[keys[1]] - spec[1]) / spec[1]
  # exclude key components
  denom_nonkey = np.delete(denom, keys)
  F_i_nonkey = np.delete(F_i, keys)
  D_i_nonkey = F_i_nonkey * denom_nonkey / (1. + denom_nonkey)
  B_i_nonkey = F_i_nonkey / (1. + denom_nonkey)
  tops = D_i_nonkey > B_i_nonkey
  D_i_nonkey[tops] = F_i_nonkey[tops] - B_i_nonkey[tops]
  B_i_nonkey[~tops] = F_i_nonkey[~tops] - D_i_nonkey[~tops]
  # reinsert key components from spec
  D_i = np.insert(D_i_nonkey, [keys[0], keys[1]-1], (spec[0], F_i[keys[1]] - spec[1]))
  B_i = np.insert(B_i_nonkey, [keys[0], keys[1]-1], (F_i[keys[0]] - spec[0], spec[1]))
  return common.SolutionObj(D_i = D_i, D_i = B_i, N_min = N_min)

def winn_coeff_est(K_i: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
  '''
  Estimates the zeta coefficient and iota power of binary equalibrium pairs for use in the Winn equation.
  
  Parameters:
  -----------
  K_i : NDArray
    Equilibrium constants of all components at two or more points in the distillation column. Shape must be M x N, M >= 2.
      ex) np.array([[K_1_D, K_2_D, K_3_D], [K_1_F, K_2_F, K_3_F], [K_1_B, K_2_B, K_3_B]])
  
  Returns:
  ----------
  phi : npt.NDArray
    Exponent of the Winn K-transform function.
  logzeta : npt.NDArray
    Log10 of the coefficient of the Winn K-transform function.
  '''
  K_i = np.log10(np.atleast_2d(K_i))
  points = np.insert(K_i.flatten("F"), np.arange(K_i.size, dtype=int)+1, np.tile(K_i[:, 0], K_i[0].size)).reshape(-1, len(K_i), 2)
  lines = [common.point_conn(v[:-1, :], v[1:, :], avgmode=True) for v in points]
  phi = np.array([line.m for line in lines]).astype(float)
  logzeta = np.array([line.b for line in lines]).astype(float)
  return phi, logzeta

def winn_plates(K_i: npt.NDArray, D_i: npt.NDArray, B_i: npt.NDArray, keys: tuple[int, int]) -> common.SolutionObj[npt.NDArray, npt.NDArray, float]:
  '''
  Calculates the minimum number of stages for a multi-component distillation tower using the Winn equation, alongside a graphical transformation.
  
  Parameters:
  -----------
  K_i : NDArray
    Equilibrium constants of all components at two or more points in the distillation column. Shape must be M x N, M >= 2.
      ex) np.array([[K_1_D, K_2_D, K_3_D], [K_1_F, K_2_F, K_3_F], [K_1_B, K_2_B, K_3_B]])
  D_i : NDArray
    Molar flowrates of all components in the distilate stream.
  B_i : NDArray
    Molar flowrates of all components in the bottoms stream. 
  keys : tuple[int, int]
    Indexes of the Light Key species and Heavy Key species in the feed array.
  
  Returns:
  ----------
  phi : npt.NDArray
    Exponent of the Winn K-transform function.
  logzeta : npt.NDArray
    Log10 of the coefficient of the Winn K-transform function.
  N_min : float
    Minimum number of stages of a multi-component distillation tower.
  '''
  K_i = np.atleast_2d(K_i)
  D_i = np.atleast_1d(D_i); B_i = np.atleast_1d(B_i)
  x_D = D_i / D_i.sum(); x_B = B_i / B_i.sum()
  phi, logzeta = winn_coeff_est(K_i)
  N_min = np.log10( (x_B[keys[1]] / x_D[keys[1]])**phi * x_D[keys[0]] / x_B[keys[0]] ) / logzeta
  return common.SolutionObj(phi = phi, logzeta = logzeta, N_min = N_min)

def winn_feed_split(K_i: npt.NDArray, F_i: npt.NDArray, D_i: npt.NDArray, B_i: npt.NDArray, keys: tuple[int, int], spec: tuple[float, float]) -> common.SolutionObj[npt.NDArray, npt.NDArray, float]:
  '''
  Calculates the molar flowrates of non-key components in the distillate and bottoms streams of a multi-component distillation using the Winn equations. 
  
  Parameters:
  -----------
  K_i : NDArray
    Equilibrium constants of all components at two or more points in the distillation column. Shape must be M x N, M >= 2.
      ex) np.array([[K_1_D, K_2_D, K_3_D], [K_1_F, K_2_F, K_3_F], [K_1_B, K_2_B, K_3_B]])
  F_i : NDArray
    Molar flowrates of all components in the feed stream.
  D_i : NDArray
    Molar flowrates of all components in the distilate stream.
  B_i : NDArray
    Molar flowrates of all components in the bottoms stream.
  keys : tuple[int, int]
    Indexes of the Light Key species and Heavy Key species in the feed array.
  spec : tuple[float, float]
    Required molar flowrate of the Light Key species in the distilate and Heavy Key species in the bottoms.
  
  Returns:
  ----------
  D_i : NDArray
    Improved molar flowrates of all components in the distilate stream.
  B_i : NDArray
    Improved molar flowrates of all components in the bottoms stream. 
  N_min : float
    Minimum number of stages for a multi-component distillation tower.
  '''
  K_i = np.atleast_2d(K_i)
  F_i = np.atleast_1d(F_i); D_i = np.atleast_1d(D_i); B_i = np.atleast_1d(B_i)
  phi, logzeta, N_min = winn_plates(K_i, D_i, B_i, keys).unpack()
  denom = (spec[1] / (F_i[keys[1]] - spec[1]))**phi * (B_i.sum() / D_i.sum())**(1. - phi) / (10.**logzeta*N_min)
  # exclude key components
  denom_nonkey = np.delete(denom, keys)
  F_i_nonkey = np.delete(F_i, keys)
  D_i_nonkey = F_i_nonkey / (1. + denom_nonkey)
  B_i_nonkey = F_i_nonkey / (1. + 1. / denom_nonkey)
  tops = D_i_nonkey > B_i_nonkey
  D_i_nonkey[tops] = F_i_nonkey[tops] - B_i_nonkey[tops]
  B_i_nonkey[~tops] = F_i_nonkey[~tops] - D_i_nonkey[~tops]
  # reinsert key components from spec
  D_i = np.insert(D_i_nonkey, [keys[0], keys[1]-1], (spec[0], F_i[keys[1]] - spec[1]))
  B_i = np.insert(B_i_nonkey, [keys[0], keys[1]-1], (F_i[keys[0]] - spec[0], spec[1]))
  return common.SolutionObj(D_i = D_i, B_i = B_i, N_min = N_min)

def underwood_type1(a_i_hk: npt.NDArray, F_i: npt.NDArray, D_i: npt.NDArray, keys: tuple[int, int], spec: tuple[float, float], psi: float) -> common.SolutionObj[npt.NDArray, npt.NDArray, float]:
  '''
  Calculates the minimum reflux ratio and component distilate streams of a Type I distilation column (full component distribution) using the Underwood equations.
  
  Parameters:
  -----------
  a_i_hk : NDArray
    Relative volatility of each compound to the heavy key compound at the final distilate plate, average column conditions, and final reboiler plate (unitless).
  F_i : NDArray
    Molar flowrates of all components in the feed stream.
  D_i : NDArray
    Molar flowrates of all components in the distilate stream.
  keys : tuple[int, int]
    Indexes of the Light Key species and Heavy Key species in the feed array.
  spec : tuple[float, float]
    Required molar flowrate of the Light Key species in the distilate and Heavy Key species in the bottoms.
  psi : float
    Vapor to liquid feed ratio (unitless).
  
  Returns:
  ----------
  D_i : NDArray
    Improved molar flowrates of all components in the distilate stream.
  typeI : NDArray
    If a component distributes across both the distilate and bottoms outflow streams (True), a distilation column is Type II if any component does not distribute (False).
  R_min : float
    Minimum reflux ratio of the a distilation column as a Type I System.
  '''
  a_i_hk = np.atleast_2d(a_i_hk)
  F_i = np.atleast_1d(F_i); D_i = np.atleast_1d(D_i)
  F_liq_i = (1. - psi) * F_i
  brack = D_i[keys[0]] / F_liq_i[keys[0]] - a_i_hk[1, keys[0]] * D_i[keys[1]] / F_liq_i[keys[1]]
  L_inf = brack * F_liq_i.sum() / (a_i_hk[1, keys[0]] - 1.)
  R_min = L_inf / D_i.sum()
  # distribution
  a_i_hk_nonkey = np.delete(a_i_hk, keys, axis=1)
  lkhalf = spec[0] * (a_i_hk_nonkey[1] - 1.) / ((a_i_hk[1, keys[0]] - 1.) * F_liq_i[keys[0]])
  hkhalf = D_i[keys[1]] * (a_i_hk[1, keys[0]] - a_i_hk_nonkey[1]) / ((a_i_hk[1, keys[0]] - 1.) * F_liq_i[keys[1]])
  D_i_nonkey = (lkhalf + hkhalf) * np.delete(F_liq_i, keys)
  D_i = np.insert(D_i_nonkey, [keys[0], keys[1]-1], (spec[0], F_i[keys[1]] - spec[1]))
  typeI = np.all((D_i / F_i > 0., D_i / F_i < 1.) , axis=0)
  return common.SolutionObj(D_i = np.minimum(np.maximum(D_i, 0.), F_i), typeI = typeI, R_min = R_min)

# I am trying to make sense of this too. Trying to figure out how to get Theta -> Flow rates
def quanderwood_type2(x_i_F: npt.NDArray, a_i_hk_F: npt.NDArray, typeI: npt.NDArray, psi: float):
  x_i_F = np.atleast_1d(x_i_F)
  a_i_hk_F = np.atleast_1d(a_i_hk_F)
  typeI = np.atleast_1d(typeI)
  theta_range = (np.min(a_i_hk_F[typeI]), np.max(a_i_hk_F[typeI]))
  theta = np.array([])
  
  def err(theta):
    return psi - np.sum(a_i_hk_F * x_i_F / (a_i_hk_F - theta))
  
  for i in np.linspace(theta_range[0], theta_range[1], 1000):
    j = common.root_newton(err, i) 
    if j < theta_range[1] and j > theta_range[0]:
      theta = np.append(theta,j)
  return np.unique(np.round(theta,3))

def underwood_type2(a_i_hk: npt.NDArray, x_F: npt.NDArray, D_i: npt.NDArray, typeI: npt.NDArray, keys: tuple[int, int], psi: float) -> common.SolutionObj[npt.NDArray, float]:
  '''
  Calculates the minimum reflux ratio and component distilate streams of a Type II distilation column (incomplete component distribution) using the Underwood equations.
  
  Parameters:
  -----------
  a_i_hk : NDArray
    Relative volatility of each compound to the heavy key compound at the final distilate plate, average column conditions, and final reboiler plate (unitless).
  x_F : NDArray
    Component mole fractions of the liquid mixture in the feed stream (unitless). Must sum to 1.
  D_i : NDArray
    Molar flowrates of all components in the distilate stream.
  typeI : NDArray
    If a component distributes across both the distilate and bottoms outflow streams (True), a distilation column is Type II if any component does not distribute (False).
  psi : float
    Vapor to liquid feed ratio (unitless).
    
  Returns:
  ----------
  D_i : NDArray
    Improved molar flowrates of all components in the distilate stream.
  R_min : float
    Minimum reflux ratio of the a distilation column as a Type I System.
  '''
  a_i_hk = np.atleast_2d(a_i_hk)
  x_F = np.atleast_1d(x_F); D_i = np.atleast_1d(D_i)
  typeI = np.atleast_1d(typeI)
  tIa = a_i_hk[1][typeI]
  thetaranges = np.linspace(tIa[:-1] - .001, tIa[1:] + .001, 100).flatten("F")
  thetasets = np.vstack(np.lib.stride_tricks.sliding_window_view(thetaranges, 2))
  
  def err(theta):
    return psi - np.sum(a_i_hk[1] * x_F / (a_i_hk[1] - np.c_[theta.flatten("A")]), axis=1, keepdims=True).reshape(-1, 2)
  
  theta, _, _ = common.err_reduc_iterative(err, thetasets, bounds=tIa, ceil=tIa.max(), floor=tIa.min())
  theta = np.array([np.average(thet) for thet in common.array_boundsplit(theta, tIa)])
  
  # assumes distrib organized like [typeII D only, (typeI, LK, typeI, HK, typeI), typeII B only]
  ngroup = [len(group) for group in common.array_boundsplit(typeI)]
  eq2 = a_i_hk[0, :ngroup[:1].sum()] / (a_i_hk[0, :ngroup[:1].sum()] - np.c_[theta])
  ind_invar = np.append(np.arange(ngroup[0]), keys)
  coeff = np.full((theta.size + 1, typeI.sum()), -1.)
  coeff[:-1, :-2] = np.delete(eq2, ind_invar)
  coeff[-1, :-2] = 1.; coeff[-1, -1] = 0.
  consts = np.zeros(theta.size + 1)
  consts[:-1] = eq2[ind_invar].sum(axis=1)
  consts[-1] = D_i[ind_invar].sum()
  variants = np.linalg.solve(coeff, consts) # please work :3
  D_i[ngroup[0]:ngroup[:1].sum()] = variants[:-2]
  R_min = variants[-1] / variants[-2]
  return common.SolutionObj(D_i = D_i, R_min = R_min)

def gilliland(N_min: float, R_min: float, R: float) -> float:
  '''
  Calculates the number of real trays required to operate a multicomponent distilation column.
  
  Parameters
  ----------
  N_min : float
    Minimum number of trays required to operate the multicomponent distilation column. Usually calculated from the Fenske equation .
  R_min : float
    Minimum reflux ratio required to operate the multicomponent distilation column.
  R : float
    Reflux ratio of the multicomponent distilation column.
  
  Returns:
  ----------
  ideal_stages : float
    Ideal number of stages required to operate the column.
  '''
  X = (R - R_min) / (R + 1)
  Y = 1 - np.exp((1. + 54.4 * X) / (11. + 117.2 * X) * ((X - 1.) / np.sqrt(X)))
  # TODO #32 more comprehensive checks for all 6 gilliland correlation limitations
  if R_min > 0.53 or R_min < 0.53 or N_min <3.4 or N_min > 60.3:
    raise Exception('Gilliland correlation is not valid in this case!')
  return (N_min + Y) / (1. - Y)

def kirkbride(x_F: npt.NDArray, D_i: npt.NDArray, B_i: npt.NDArray, keys: tuple[int, int], actual_trays: float) -> tuple[float, float]:
  '''
  Calculates the location of the feed tray in a multicomponent distilation column using the Kirkbride equation.
  
  Parameters:
  -----------
  x_F : NDArray
    Component mole fractions of the liquid mixture in the feed stream (unitless). Must sum to 1.
  D_i : NDArray
    Molar flowrates of all components in the distilate stream.
  B_i : NDArray
    Molar flowrates of all components in the bottoms stream.
  keys : tuple[int, int]
    Indexes of the Light Key species and Heavy Key species in the feed array.
  actual_trays : float
    Number of actual trays in a multicomponent distilation column, having already accounted for the reboiler, consenser, and expected plate efficiency.
    
  Returns:
  ----------
  trays_D : float
    Number of rectifying trays in a multicomponent distilation column.
  trays_S : float
    Number of stripping trays in a multicomponent distilation column, where the first tray is the feed tray.
  '''
  x_F = np.atleast_1d(x_F)
  D_i = np.atleast_1d(D_i); B_i = np.atleast_1d(B_i)
  x_D = D_i / D_i.sum(); x_B = B_i / B_i.sum()
  chunk = ( x_F[keys[1]] * x_B[keys[0]]**2 * B_i.sum() / (x_F[keys[0]] * x_D[keys[1]]**2 * D_i.sum()) )**.206
  trays_D = actual_trays + chunk / (1. + chunk)
  return trays_D, actual_trays - trays_D

def multicomp_heat_dut(heatvap_i: npt.NDArray, F_i: npt.NDArray, D_i: npt.NDArray, B_i: npt.NDArray, R: float, psi: float) -> tuple[float, float]:
  '''
  Calculates the heat duties of the condenser and reboiler in a multicomponent distilation column.
  
  Parameters:
  -----------
  heatvap_i : NDArray
    Heat of vaporization (or condensation) of all components.
  F_i : NDArray
    Molar flowrates of all components in the feed stream.
  D_i : NDArray
    Molar flowrates of all components in the distilate stream.
  B_i : NDArray
    Molar flowrates of all components in the bottoms stream.
  keys : tuple[int, int]
    Indexes of the Light Key species and Heavy Key species in the feed array.
  actual_trays : float
    Number of actual trays in a multicomponent distilation column, having already accounted for the reboiler, consenser, and expected plate efficiency.
    
  Returns:
  ----------
  Q_cond : float
    Heat duty of the condenser.
  Q_reb : float
    Heat duty of the reboiler.
  '''
  heatvap_i = np.atleast_1d(heatvap_i)
  F_i = np.atleast_1d(F_i); D_i = np.atleast_1d(D_i); B_i = np.atleast_1d(B_i)
  V_D = D_i.sum() * (1. + R)
  V_S = V_D - F_i.sum() * psi
  Q_cond = V_D * np.sum(heatvap_i * D_i / D_i.sum())
  Q_reb = V_S * np.sum(heatvap_i * B_i / B_i.sum())
  return Q_cond, Q_reb

def column_design_full_est(ant_coeff: npt.NDArray, F_i: npt.NDArray, MW: npt.NDArray, Tc: npt.NDArray, Pc: npt.NDArray, heatvap_i: npt.NDArray, 
                           keys: tuple[int, int], spec: tuple[float, float],
                           Rmin_mult: float = 1.2, tray_eff: float = .85, T_D: float = 312.15, T_decomp: float | None = None, numplates: float | None = None, vacuumColumn: bool = False, decompSafeFac: float = .5, tol: float = .001):
  '''
  Calcualtes the pressure across a distilation column.
  
  Parameters:
  -----------
  ant_coeff : NDArray
    Coefficients for the Antoine Equation of State for all components (unitless). Shape must be N x 3.
  F_i : NDArray
    Molar flowrates of all components in the feed stream. Length must be N.
  MW : NDArray
    Molecular weight in g/mol (grams per mole) of each input feed species. Length must be N.
  Tc : npt.NDArray
    Critical temperature of all components in K (Kelvin). Length must be N.
  Pc : npt.NDArray
    Critical pressure of all components in atm (atmospheres). Length must be N.
  heatvap_i : NDArray
    Heat of vaporization (or condensation) of all components.
  keys : tuple[int, int]
    Indexes of the Light Key species and Heavy Key species in the feed array.
  spec : tuple[float, float]
    Required molar flowrate of the Light Key species in the distilate and Heavy Key species in the bottoms.
  Rmin_mult : float
    Factor by which to excede the minimum reflux ratio, R_min (unitless). Typical reflux ratios are between 1.05 and 1.3 times Rmin. Bounded (1, inf).
  tray_eff : float
    Performance efficiency of a tray relative to its theorhetical perfection. 
  T_D : float
    Temperature of the distillate liquid in the reflux drum in K (Kelvin). Assumes 49 C (Celcius) == 312.15 Kelvin (K) by default.
  T_decomp : float
    Temperature of decomposition of the bottoms product in K (Kelvin).
  numplates : float
    Number of plates in the distilation column, if known.
  vacuumColumn : bool
    If the distilation column is operating below ambient pressure.
  decompSafeFac : float
    Ratio of the maximum reboiler temperature to the bottom product's decompostion temperature.
  
  Returns:
  -----------
  T_and_P : NDArray
    Temperature in K (Kelvin) and pressure in psia (absolute pounds per square inch) pairs for the top, average, and bottom of the distilation column.
  condenserType : str
    Type of condenser that ought to be used at the calculated distilate pressure.
  '''
  x_F = F_i / F_i.sum()
  D_i, B_i = multicomp_feed_split_est(F_i, MW, keys, spec)
  psi = .5; N_min = numplates
  D_i_old = np.full_like(D_i, np.NaN)
  while not ((np.abs(D_i - D_i_old)) < tol).all():
    while not ((np.abs(D_i - D_i_old)) < tol).all():
      D_i_old = np.copy(D_i)
      T_and_P, condenserType = multicomp_column_cond(ant_coeff, D_i, B_i, T_D, T_decomp, N_min, vacuumColumn, decompSafeFac)
      K_i = props.k_wilson(ant_coeff, Tc, Pc, T_and_P)
      psi = psi_solver(x_F, K_i[1], psi)
      a_i_hk = K_i / np.c_[K_i[:, keys[1]]]
      try:
        D_i, B_i, N_min = fenske_feed_split(a_i_hk, F_i, D_i, B_i, keys, spec).unpack()
      except:
        D_i, B_i, N_min = winn_feed_split(K_i, F_i, D_i, B_i, keys, spec).unpack()
    D_i, typeI, R_min = underwood_type1(a_i_hk, F_i, D_i, keys, spec, psi).unpack()
    D_i, R_min = underwood_type2(a_i_hk, x_F, D_i, typeI, keys, psi).unpack()
  
  R = R_min * Rmin_mult
  ideal_stages = gilliland(N_min, R_min, R)
  ideal_trays = ideal_stages - 1. if condenserType == "Total Condenser" else 2.
  actual_trays = ideal_trays / tray_eff
  trays_D, trays_S = kirkbride(x_F, D_i, B_i, keys, actual_trays)
  multicomp_heat_dut(heatvap_i, F_i, D_i, B_i, R, psi)
  return common.SolutionObj(Rmin = R_min, R = R, trays_D = trays_D, trays_S = trays_S)
