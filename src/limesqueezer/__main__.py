#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Entrypoint module, in case you use `python -m limesqueezer`.


Why does this file exist, and why __main__? For more info, read:

- https://www.python.org/dev/peps/pep-0338/
- https://docs.python.org/2/using/cmdline.html#cmdoption-m
- https://docs.python.org/3/using/cmdline.html#cmdoption-m
"""

###═════════════════════════════════════════════════════════════════════
### IMPORT


from scipy import interpolate
import sys
import os
import pathlib
import matplotlib.pyplot as plt
import time
import numpy as np

sys.path.insert(1,str(pathlib.Path(__file__).parent.absolute()))
import API as lc

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
elif sys.argv[1] == 'debug':
    import plotters
    debugplot = plotters.Debug()
    debugplot.run()
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
    # if verbosity>0: 
    #     text = 'Length of compressed array\t%i'%len(x_c)
    #     text += '\nCompression factor\t%.3f %%' % (100*len(x_c)/len(x))
    #     if is_timed: text += '\nCompression time\t%.1f ms' % (t*1e3)
    #     print(text)
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

if is_show: plt.show()