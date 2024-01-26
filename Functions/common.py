import numpy as np
import pandas as pd

def testvars():
  TEST_numb_comp = 3
  TEST_mol_frac = np.array([.25, .40, .35])
  TEST_x = np.array([.25, .40, .35])
  TEST_y = np.array([0.576857, 0.312705, 0.102059])
  TEST_ant_coeff = np.array([[6.82973, 813.2, 248.],
                            [6.83029, 945.90, 240.],
                            [6.85221, 1064.63, 232.]])
  TEST_P = 10342.95

  # numb_comp, x, ant_coeff, P = common.testvars()
  return TEST_numb_comp, TEST_mol_frac, TEST_ant_coeff, TEST_P

def input_mol_frac(phase = "l"):
  # promt for component quantity
  numb_comp = int(input ('Total Number of Components? : '))
  print("")
  v = np.zeros(numb_comp)

  # ask for input of the incoming feed composition
  abbrev = {
    "s": "Solid",
    "l": "Liquid",
    "v": "Vapor"
  }
  print(f" --- Composition of the {abbrev[phase]} Feed --- ")
  # print('---')
  for i,item in enumerate(v):
    v[i] = float(input(f'Enter Mol Fraction of Component {i + 1} : '))
  print('')
  return numb_comp, v

def input_Kfac(numb_comp):
  k = np.zeros(numb_comp)
  for i, _ in enumerate(k):
    k[i] = float(input(f'Enter K Factor of Component {i + 1} : '))
  print('\n')
  return k

def input_antoine(numb_comp, numb_coeff = 3):
  ant_coeff = np.zeros((numb_comp, numb_coeff)) 
  #numb_coeff must be <= 25
  coeffs = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  for i, components in enumerate(ant_coeff):
    for j, _ in enumerate(components):
      ant_coeff[i, j] = float(input(f'Enter Antoine Coefficient {coeffs[j]} for Component {i + 1} : '))
    print('')
  return ant_coeff

def input_P(vessel = "fd"):
  abbrev = {
    "fd": "Flash Drum",
    "t": "Tank",
  }
  P = float(input(f'Enter the pressure of the {abbrev[vessel]} in mmHg : '))
  print('')
  return P

def antoine_T(v, P):
  # T in C, P in mmHg
  v = np.atleast_1d(v); P = np.atleast_1d(P)

  def antoine_T_3(v, P):
    return (-v[:, 1] / (np.log10(P) - np.r_[v[:, 0]])) - v[:, 2]
  
  if len(v[1, :]) == 3:
    return antoine_T_3(v, P)
  else:
    T = np.zeros((len(v), len(P)))
    for i_comp, ant_coeff in enumerate(v):
      for j_temp, press in enumerate(P):
        a, b, c = ant_coeff
        T[i_comp, j_temp] = -b / (np.log10(press) - a) - c
    return T

def antoine_P (v, T):
  # T in C, P in mmHg
  v = np.atleast_1d(v); T = np.atleast_1d(T)

  def antoine_P_3(v, T):
    return 10 ** (np.c_[v[:, 0]] - np.c_[v[:, 1]] / (T + np.c_[v[:, 2]]))
  
  if len(v[1, :]) == 3:
    return antoine_P_3(v, T)
  else:
    P = np.zeros((len(v), len(T)))
    for i_comp, ant_coeff in enumerate(v):
      for j_temp, temp in enumerate(T):
        a, b, c = ant_coeff
        P[i_comp, j_temp] = 10 ** (a - (b) / (temp + c))
    return P

def lin_estimate_error(x_pair, y_pair):
  return x_pair[0] - y_pair[0] * ((x_pair[1]-x_pair[0])/(y_pair[1]-y_pair[0]))
