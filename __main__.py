#!/usr/bin/python3
# -*- coding: utf-8 -*-
###═════════════════════════════════════════════════════════════════════
### IMPORT
import API as lc
from scipy import interpolate
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
        exit()
    #───────────────────────────────────────────────────────────────────────
    elif sys.argv[1] == 'cwd':
        print(os.getcwd())
if sys.argv[1] == 'tests':
    import tests
    exit()

path_home = pathlib.Path(__file__).parent.absolute()

path_figures = path_home / 'figures'
###═════════════════════════════════════════════════════════════════════
n_data = int(float(sys.argv[1]))
ytol = float(sys.argv[2])
mins = int(float(sys.argv[3]))
b = int(float(sys.argv[4]))
data = lc.Data(n_data=n_data,b=b)
data.x_compressed, data.y_compressed = lc.compress(data.x,data.y,
                                                   ytol=ytol, mins = mins,
                                                   verbosity = verbosity,
                                                   is_timed = is_timed)
print(data.x_compressed[-1])
y0 = np.array([data.y[0],data.y[0]+data.x[0]])

with lc.Compressed(data.x[0], y0,ytol=ytol, mins=mins) as compressed:
    plt.ion()
    
    fig, ax = plt.subplots()
    ax.set_title("interactive test")
    ax.set_xlabel("x")
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)
    ax.plot(data.x,data.y)
    ln = ax.plot(compressed.x,compressed.y[:,0],'-o')
    t_start = time.perf_counter()
    xlim = 0
    for x,y in zip(data.x,data.y):
        compressed(x,np.array([y, y+x]))
        # print(compressed.x)
        # print(compressed.y)
        if x>xlim:
            xlim += 0.01
            if type(ln) == list:
                for index, line in enumerate(ln):
                    line.set_xdata(compressed.x)
                    line.set_ydata(compressed.y[:,index])
            else:
                ln.set_xdata(compressed.x)
                ln.set_ydata(compressed.y)
            fig.canvas.draw()
    plt.show()
    time.sleep(2)
print("compression time",time.perf_counter()-t_start)
# # print(compressed)

# # for x,y in zip(compressed.x,data.x_compressed):
# #     print(x,y)
plt.show()
print(len(compressed))
# # print(compressed.y[:,0])
# print(compressed.x[-1])
# print(data.x[-1])
# print(data.x_compressed - compressed.x)
#───────────────────────────────────────────────────────────────────
if is_plot:
    plt.figure()
    plt.plot(data.x,data.y)
    plt.plot(data.x_compressed,data.y_compressed,'-o')
    title = 'LSQ compressed data'
    plt.title(title)
    if is_save: plt.savefig(path_figures/(title+'.pdf'), bbox_inches='tight')
    #───────────────────────────────────────────────────────────────────
    data.make_lerp()
    print(data.NRMSE)
    print('maxres: function',max(abs(data.residuals)))
    plt.figure()
    plt.plot(data.x,data.residuals)
    title = 'LSQ compressed residuals'
    plt.title(title)
    if is_save: plt.savefig(path_figures/(title+'.pdf'), bbox_inches='tight')

    lerp = interpolate.interp1d(compressed.x,compressed.y[:,0], assume_sorted=True)
    residuals = lerp(data.x) - data.y
    print('max relative residual',np.amax(np.abs(residuals))/ytol)

    plt.figure()
    plt.plot(compressed.x,compressed.y[:,0],'-o')
    title = 'Loooped compressed data'
    plt.title(title)
    if is_save: plt.savefig(path_figures/(title+'.pdf'), bbox_inches='tight')

    plt.figure()
    plt.plot(data.x,residuals,'-')
    title = 'Loooped compressed residuals'
    plt.title(title)
    if is_save: plt.savefig(path_figures/(title+'.pdf'), bbox_inches='tight')
#───────────────────────────────────────────────────────────────────
# data2 = lc.Data(n_data=n_data,b=b)
# data2.simplecompress(ytol=ytol,mins = mins,verbosity=verbosity)

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
# data3.fastcompress(ytol=ytol, mins = mins*10,verbosity = verbosity)

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
# indices = lc.fastcompress(data3.x, data3.y, ytol=ytol, mins = mins)

# data3.x_compressed, data3.y_compressed = data3.x[indices], data3.y[indices]
# t = time.perf_counter()-t_start
# print('Compression time\t%.3f ms' % (t*1e3))
# print('Length of compressed array\t%i'%len(xce))
# compression_residual = 1 - len(xce)/len(data3.x)
# print('Compression factor\t%.3f %%' % (compression_residual*1e2))

# data3.make_lerp()
# tol = abs(data3.residual())-data3.ytol
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