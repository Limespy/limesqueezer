#!/usr/bin/python3
# -*- coding: utf-8 -*-
###═════════════════════════════════════════════════════════════════════
### IMPORT
import API as lc

import sys
import os
import pathlib
import matplotlib.pyplot as plt

import numpy as np
import math

helpstring = "No arguments given"

is_verbose = "--verbose" or "-v" in sys.argv
is_plot = "--plot" in sys.argv
is_save = "--save" in sys.argv
is_show = "--show" in sys.argv

if len(sys.argv)==1:
    print(helpstring)
    exit()
else:
    path_cwd = pathlib.Path(os.getcwd()).absolute()
    if is_verbose: print("Selected path is:\n\t%s" % path_cwd)

    #───────────────────────────────────────────────────────────────────────
    elif sys.argv[1] == "sandbox":
        args = sys.argv[2:]
        import sandbox
    #───────────────────────────────────────────────────────────────────────
    elif sys.argv[1] == "cwd":
        print(os.getcwd())


path_home = pathlib.Path(__file__).parent.absolute()

path_figures = path_home / "figures"


def reference(x):
    # return np.sin(x) + 2
    return 2/(x-2) + 2


compression = lc.Compression() 

errtol = float(sys.argv[2])

print(len(compression.y_data))

print(compression.err_n(1520)-errtol)

# x1, y1, x2, y2 = lc.reach(lambda x: lc.err_n(x,errtol), -errtol)

# n_lim = lc.int_bisect(lambda x: lc.err_n(x,errtol), x1, y1, x2, y2)+2
# print(n_lim)
# print(lc.err_max_LSQ(x_input[:n_lim],y_input[:n_lim])-errtol)
# print(lc.err_max_LSQ(x_input[:n_lim+1],y_input[:n_lim+1])-errtol)

# err_budget = np.array([lc.err_n(n,errtol) for n in range(80)])

# plt.figure()
# plt.plot(range(80),err_budget)
# plt.plot([0,80],[0,0])
# plt.plot([70,70],[-errtol,+errtol])
# plt.plot([69,69],[-errtol,+errtol])

def plot():
    pass


# Prediction heuristic 2
# Function is approximately a*x^2 - atol
# Fitting a*x1^2 - atol = y1 -> a = (y1-atol)/x1^2
# Root a*x^2 -atol = 0 -> x = sqrt(atol/a) 
# Root a*x^2 -atol = 0 -> Prediction_1 = x1 / sqrt(y1/atol-1) 


