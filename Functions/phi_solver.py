import numpy as np
import pandas as pd
import common

def phi_solver(x, K, phi, tol = 0.01):
    def f(phi):
        return np.sum( (x * (1- K)) / (1 + phi * (K - 1)) )
    def f_prime(phi):
        return np.sum( (x * (1 - K)**2) / (1 + phi * (K - 1))**2)
    def phi_next(phi):
        return phi - (f(phi) / f_prime(phi))
    def error(phi):
        return (phi_next(phi) - phi) / phi
    
    table = np.array([['Phi', 'f', 'f_prime', 'phi_next', 'error']], dtype='object')
    
    i = 0
    while tol < np.abs(f(phi)) :
        table = np.concatenate((table, [[phi,f(phi), f_prime(phi), phi_next(phi), error(phi)]] ))
        phi = phi_next(phi)
        i += 1

    table = np.concatenate((table, [[phi,f(phi), f_prime(phi), phi_next(phi), error(phi)]] )) # This must be done one more time for the final iteration
    return phi, i, table
    
numb_comp, x = common.input_mol_frac()

k = common.input_Kfac(numb_comp)

phi_i = float(input('Input an initial guess for phi : '))

phi, iterations, table = phi_solver(x, k, phi_i)
print(' ---- ')
print(f'The converged value of phi is : {phi}')
print(f'This was done in {iterations} iterations .')
print('----')
print(pd.DataFrame(table))

input('Exit? Y/N ')    