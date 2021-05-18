#!/usr/bin/python3
# -*- coding: utf-8 -*-
###═════════════════════════════════════════════════════════════════════
### IMPORT
import API as lc

import sys
import os
import pathlib
import matplotlib.pyplot as plt
import time
import numpy as np

helpstring = 'No arguments given'

verbosity = 1 if ('--verbose' in sys.argv or '-v' in sys.argv) else 0
is_timed = ('--timed' in sys.argv or '-t' in sys.argv)
is_plot = '--plot' in sys.argv
is_save = '--save' in sys.argv
is_show = '--show' in sys.argv
if len(sys.argv)==1:
    print(helpstring)
    exit()
else:
    path_cwd = pathlib.Path(os.getcwd()).absolute()
    if verbosity>0: print('Selected path is:\n\t%s' % path_cwd)

    #───────────────────────────────────────────────────────────────────────
    elif sys.argv[1] == 'sandbox':
        args = sys.argv[2:]
        import sandbox
    #───────────────────────────────────────────────────────────────────────
    elif sys.argv[1] == 'cwd':
        print(os.getcwd())


path_home = pathlib.Path(__file__).parent.absolute()

path_figures = path_home / 'figures'
###═════════════════════════════════════════════════════════════════════
n_data = int(float(sys.argv[1]))
atol = float(sys.argv[2])
mins = int(float(sys.argv[3]))
b = int(float(sys.argv[4]))
data = lc.Data(n_data=n_data,b=b)
data.x_compressed, data.y_compressed = lc.compress(data.x,data.y,
                                                   atol=atol, mins = mins,
                                                   verbosity = verbosity,
                                                   is_timed = is_timed)
#───────────────────────────────────────────────────────────────────
if is_plot:
    plt.figure()
    plt.plot(data.x,data.y)
    plt.plot(data.x_compressed,data.y_compressed,'-o')
    title = 'LSQ compressed data'
    plt.title(title)
    if is_save: plt.savefig(path_figures/(title+'.png'), bbox_inches='tight')
    #───────────────────────────────────────────────────────────────────
    data.make_lerp()
    print(data.NRMSE)
    print(data.covariance)

    plt.figure()
    plt.plot(data.x,data.residuals)
    title = 'LSQ compressed residuals'
    plt.title(title)
    if is_save: plt.savefig(path_figures/(title+'.png'), bbox_inches='tight')
#───────────────────────────────────────────────────────────────────
# data2 = lc.Data(n_data=n_data,b=b)
# data2.simplecompress(atol=atol,mins = mins,verbosity=verbosity)

# if is_plot:
#     plt.figure()
#     plt.plot(data2.x,data2.y)
#     plt.plot(data2.x_compressed,data2.y_compressed,'-o')
#     title = 'Simple compressed data'
#     plt.title(title)
#     if is_save: plt.savefig(path_figures/(title+'.png'), bbox_inches='tight')
#     #───────────────────────────────────────────────────────────────────
#     data2.make_lerp()
#     plt.figure()
    
#     plt.plot(data2.x,data2.residuals())
#     title = 'Simple compressed residuals'
#     plt.title(title)
#     if is_save: plt.savefig(path_figures/(title+'.png'), bbox_inches='tight')
# #───────────────────────────────────────────────────────────────────
# data3 = lc.Data(n_data=n_data,b=b)
# data3.fastcompress(atol=atol, mins = mins*10,verbosity = verbosity)

# if is_plot:
#     plt.figure()
#     plt.plot(data3.x,data3.y)
#     plt.plot(data3.x_compressed,data3.y_compressed,'-o')
#     title = 'Split compressed data'
#     plt.title(title)
#     if is_save: plt.savefig(path_figures/(title+'.png'), bbox_inches='tight')
#     #───────────────────────────────────────────────────────────────────
#     data3.make_lerp()

#     plt.figure()
#     plt.plot(data3.x,data3.residuals())
#     title = 'Split compressed residuals'
#     plt.title(title)
#     if is_save: plt.savefig(path_figures/(title+'.png'), bbox_inches='tight')

# # t_start = time.perf_counter()
# indices = lc.fastcompress(data3.x, data3.y, atol=atol, mins = mins)

# data3.x_compressed, data3.y_compressed = data3.x[indices], data3.y[indices]
# t = time.perf_counter()-t_start
# print('Compression time\t%.3f ms' % (t*1e3))
# print('Length of compressed array\t%i'%len(xce))
# compression_residual = 1 - len(xce)/len(data3.x)
# print('Compression factor\t%.3f %%' % (compression_residual*1e2))

# data3.make_lerp()
# tol = abs(data3.residual())-data3.atol
# print(max(tol))
# plt.figure()
# plt.plot(data3.x,tol)

# c = 0.5
# n = 30
# x = np.linspace(0,1,n)
# xm = 2*x-1
# y = xm*((1-c)*xm**2+c)/2+0.5
# plt.figure()
# plt.plot(x,y,'o')
# plt.plot(y,np.zeros(n),'o')

# density = 1/np.diff(y)
# dx = x[:-1] + np.diff(x)/2
# plt.figure()
# plt.plot(dx,density/(min(density)),'o')

if is_show: plt.show()