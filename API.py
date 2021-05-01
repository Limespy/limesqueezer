#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt



def reach(f, y0):
    """Finds the upper limit to interval"""

    y1 = f(1)
    if y1 > 0:
        return 0, y0, 1, y1
    else:
        print(y1)

        # Next step at 3
        y2 = f(3)
        if y2>0:
            return 1, y1, 3, y2
        
        y3 = f(7)
        if y3>0:
            return 3, y2, 7, y3

        xp1 = math.ceil(max(np.polynomial.polynomial.Polynomial.fit([0,1,3,7], [y0,y1,y2,y3], 2).roots()))

        print(xp1)
        yp1 = f(xp1)

        if yp1 > 0:
            return 7, y3, xp1, yp1
        
        xp2 = math.ceil(max(np.polynomial.polynomial.Polynomial.fit([3,7,xp1], [y2,y3,yp1], 2).roots()))
        print(xp2)
        yp2 = f(xp2)

    return prediction3, yp1, prediction4, yp2

def int_bisect(f,x1,y1,x2,y2):
    """Returns the last x where f(x)<0"""
    print(x1,x2)
    print(y1,y2)
    xdiff = x2-x1
    if xdiff > 2:
        xn = round(x1-y1/(y2-y1)*(x2-x1)) # Linear estimation
        
        if xn == x1:    # To stop repetition in close cases
            xn += 1
        elif xn == x2:
            xn -= 1

        yn = f(xn)
        if yn >0:
            x2, y2 = xn, yn
        else: 
            x1, y1 = xn, yn
        return int_bisect(f,x1,y1,x2,y2)
    
    elif xdiff == 2:
        yn = f(x1+1)
        return x1 if yn >0 else x1+1
    else:
        return x1

# Setting up the reference data

def reference(x):
    # return np.sin(x) + 2
    return 1/(x-2) + 1.5

n_input = 10e3

x_input = np.linspace(0,1,int(n_input))
y_input = reference(x_input)

def err_max_LSQ(x_data,y_data):
    fit =  np.polynomial.polynomial.Polynomial.fit(x_data,y_data,1)
    return max(abs(y_data-fit(x_data)))

errtol = 1e-4

def err_n(n):
    return err_max_LSQ(x_input[:(n+2)], y_input[:(n+2)]) -errtol

x1, y1, x2, y2 = reach(err_n, -errtol)

n_lim = int_bisect(err_n, x1, y1, x2, y2)+2
print(n_lim)
print(err_max_LSQ(x_input[:n_lim],y_input[:n_lim])-errtol)
print(err_max_LSQ(x_input[:n_lim+1],y_input[:n_lim+1])-errtol)

err_budget = np.array([err_n(n) for n in range(80)])

plt.figure()
plt.plot(range(80),err_budget)
plt.plot([0,80],[0,0])
plt.plot([70,70],[-errtol,+errtol])
plt.plot([69,69],[-errtol,+errtol])
# plt.show()


# err_allowed = 0.01
# s1 = 3
# while err_max_LSQ(x_input[:s1],y_input[:s1]) < err_allowed:
#     # Inefficient way, exponential-splitting done later
#     s1 += 1

# def err_max_lin(x_data,y_data):
#     a = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0])
#     return max(abs(y_data - (a * (x_data - x_data[0]) + y_data[0])))

# s2=3
# while err_max_lin(x_input[:s2],y_input[:s2]) < err_allowed:
#     # Inefficient way, exponential-splitting done later
#     s2 += 1



# x_1 = (x_input[s1-2] + x_input[s1-1])/2

# lerp1 = np.poly1d(np.polyfit(x_input[:(s1-1)],y_input[:(s1-1)],1))
# print(s1-1)
# print(x_1)
# print(s2-1)

# plt.figure()

# plt.plot(x_input,y_input)
# plt.plot([0,x_1],[lerp1(0),lerp1(x_1)])

# plt.show()