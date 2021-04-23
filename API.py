#!/usr/bin/python3
# -*- coding: utf-8 -*-




import numpy as np
import math


# Setting up the reference data

def reference(x):
    return np.sin(x) + 2


n_input = 10e4


x_input = np.linspace(0,10,int(n_input))

y_input = reference(x_input)

print(np.polyfit(x_input,y_input,1))