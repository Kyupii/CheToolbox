import numpy as np
import numpy.typing as npt
import pandas as pd

import common

# Old Input Code
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

# region Old print to display code
# numb_comp, x = common.input_mol_frac()
# 
# ant_coeff = common.input_antoine(numb_comp)
# 
# P = common.input_P()

# umb_comp, x, ant_coeff, P = common.testvars()

# bubbleP, Pvap, k, y, error, iters = bubble_point_iter(x, ant_coeff, P)

# print('---')
# print(f'The Bubble Temperature is {bubbleP} Celcius')
# print(f'The Error of the Calculation is {error}')
# print(f'The Calculation took {iters} iterations')
# print('---')z
# df = pd.DataFrame({'Liq Mol Fraction' : x ,
#                    'Vap Mol Fraction' : y,
#                    'Pvap (mmHg)' : Pvap,
#                    'K' : k})
# print(df)
# If we want the function to return the as is right now, or a formatted dataframe... I dont see *too* much benefit to the dataframe approach

# numb_comp, x = common.input_mol_frac()

# k = common.input_Kfac(numb_comp)

# psi_i = float(input('Input an initial guess for psi : '))
  
# table = np.array([['psi', 'f', 'f_prime', 'psi_next', 'error']], dtype='object')
# psi, iterations, table = psi_solver(x, k, psi_i)
# print(' ---- ')
# print(f'The converged value of psi is : {psi}')
# print(f'This was done in {iterations} iterations .')
# print('----')
# print(pd.DataFrame(table))

# input('Exit? Y/N ')    

# numb_comp, x = common.input_mol_frac()
# 
# ant_coeff = common.input_antoine(numb_comp)
# 
# P = common.input_P()

# numb_comp, y, ant_coeff, P = common.testvars()


# dewP, Pvap, k, x, error, iters = dew_point_iter(y, ant_coeff, P)

# print('---')
# print(f'The Bubble Temperature is {dewP} Celcius')
# print(f'The Error of the Calculation is {error}')
# print(f'The Calculation took {iters} iterations')
# print('---')
# df = pd.DataFrame({'Liq Mol Fraction' : x ,
#                    'Vap Mol Fraction' : y,
#                    'Pvap (mmHg)' : Pvap,
#                    'K' : k})
# print(df)
# endregion
