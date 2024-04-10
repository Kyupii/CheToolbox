# CheToolbox
Co-Written by Quan Phan & Ethan Molnar
## About
CheToolbox or Chemical Engineering Toolbox is a python package designed to formalize calculations learned throughout university and professional experience in code. Functions are written to be clear, concise, easy to use, and save time compared to starting from scratch. This code is written in a semi-functional programming style, with a focus on internal consistency.
## License
CheToolbox is distributed under GPL License version 3 (GPLv3)
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
x = np.array([0.1, 0.2, 0.3, 0.4]) # mole fraction of incoming stream across all components
K = np.array([4.2, 1.75, .74, .34]) # corresponding K values for all components
psi_init = 0.5 # initial guess for psi
separations.psi_solver(x, K, psi_init)
```
Returns:
```py
  psi: 0.12109169497141782
x_out: [ 7.207e-02  1.833e-01  3.098e-01  4.347e-01]
y_out: [ 3.027e-01  3.209e-01  2.292e-01  1.478e-01]
error: 0.000675806754791175
    i: 2
 ```
## Advanced Usage
```py
import numpy as np
from chetoolbox import common, separations
xf = .7 # mole fraction of feed stream
xd = .95 # mole fraction design param for distillate stream
xb = .05 # mole fraction design param for bottoms stream
q = .5 # approx. liquid fraction
Rmin_mult = 1.3 # reflux ratio multiplier (compensate for inefficiencies)

eq_points = np.array([[0.02, 0.05],
                      [0.03, 0.1],
                      [0.1, 0.3],
                      [0.2, 0.48],
                      [0.3, 0.59],
                      [0.4, 0.68],
                      [0.5, 0.75],
                      [0.59, 0.8],
                      [0.8, 0.9],
                      [0.91, 0.95]
                      ])
eq_curve = separations.eq_curve_estim(eq_points) # estimate equilibrium curve for binary mixture

liqlineH = common.point_conn((0, 15), (.95, 5)) # enthalpy curve (assumed linear) for liquid mixtures
vaplineH = common.point_conn((0, 55), (.95, 40)) # enthalpy curve (assumed linear) for vapor mixtures
Fpoint = (xf, vaplineH.eval(xf) - 30.) # enthalpy of feed

separations.ponchon_savarit_full_est(eq_curve, liqlineH, vaplineH, Fpoint, q, xd, xb, Rmin_mult, PLOTTING_ENABLED = True)
```
Returns:
```py
     tieline: <common.LinearEq object at 0x00000239FE74B710>
        Rmin: 0.5039208963545665
           R: 0.6550971652609364
  min_stages: 5.447199278007249
ideal_stages: 10.779371485763459
```
![Ponchon Savarit Diagram that Displays All Equilibrium Tielines and Both Enthalpy Lines](readme_imgs/ponchon_savarit.png "Ponchon Savarit Diagram")