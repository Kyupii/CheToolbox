# CheToolbox
Co-Written by Quan Phan & Ethan Molnar
## About
CheToolbox or Chemical Engineering Toolbox is a python package designed to formalize calculations learned throughout university and professional experience in code. Funcitons are written to be clear, conscise, easy to use, and save time compared to starting from scratch. This code is written in a semi-functional programming style, with a focus on internal consistency.
## License
CheToolbox is distributed under GPL Liscense version 3 (GPLv3)
## Dependencies
The following dependencies will be necessary for CheToolbox to build properly,
- Python >= 2.7: http://www.python.org/ (also install development files)
- SciPy >= 0.11.0: http://www.scipy.org/
- NumPy >= 1.16.0: http://www.numpy.org/
- Pandas >= 2.2.1 https://pandas.pydata.org/
## Usage
```py
import numpy as np
from chetoolbox import separations
```
```py
x = np.array([0.1, 0.2, 0.3, 0.4]) # mol fraction of incoming stream
K = np.array([4.2, 1.75, .74, .34]) # corresponding K values
psi_init = 0.5 # an initial guess for psi
```
```py
separations.psi_solver(x, K, psi_init)
```
returns
```py
{'psi': 0.12109169497141782,
 'x_out': array([0.07207241, 0.18334851, 0.30975219, 0.43474505]),
 'y_out': array([0.30270414, 0.3208599 , 0.22921662, 0.14781332]),
 'error': 0.000675806754791175,
 'i': 2}
 ```