import API as ls

import numpy as np

x_data = np.linspace(0,60,int(1e7))
y_data = np.array([np.sin(x_data*1.2*3.1416),np.sin(x_data*2*3.1416)]).T
for _ in range(100):
    xc, yc = ls.compress(x_data, y_data, tol = 1e-2)
