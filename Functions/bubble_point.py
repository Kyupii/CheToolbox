import numpy as np
import pandas as pd
import common

# numb_comp, x = common.input_mol_frac()
# 
# ant_coeff = common.input_antoine(numb_comp)
# 
# P = common.input_P()

numb_comp, x, ant_coeff, P = common.testvars()

def bubble_point_iter(x, ant_coeff, P, tol=.05):
  boil_points = common.antoine_T(ant_coeff, P)
  T = [np.max(boil_points), np.min(boil_points)]

  def calcs(T):
    Pvap = common.antoine_P(ant_coeff, T)
    k = Pvap / P
    y = np.c_[x] * k
    error =  np.sum(y, axis=0) - 1
    return Pvap, k, y, error
  
  def iter(T):
    _, _, _, error = calcs(T)
    Tnew = common.lin_estimate_error(T, error)
    error = np.abs(error)
    T[np.argmin(error)] = Tnew
    return error, T
  
  error = 10000
  i = 0
  while np.min(error) > tol:
    error, T = iter(T)
    i += 1

  bubbleP = T[np.argmin(error)]
  Pvap, k, y, error = calcs(bubbleP)
  return bubbleP, Pvap[:, 0], k[:, 0], y[:, 0], error[0], i

bubbleP, Pvap, k, y, error, iters = bubble_point_iter(x, ant_coeff, P)

print('---')
print(f'The Bubble Temperature is {bubbleP} Celcius')
print(f'The Error of the Calculation is {error}')
print(f'The Calculation took {iters} iterations')
print('---')
df = pd.DataFrame({'Liq Mol Fraction' : x ,
                   'Vap Mol Fraction' : y,
                   'Pvap (mmHg)' : Pvap,
                   'K' : k})
print(df)