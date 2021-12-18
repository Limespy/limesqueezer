#!/usr/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import math

import API as ls
import reference as ref

colors = {'AuroraBlue': '#0e1c3f',
          'AuroraCyan': '#00a0c3'}

def plot_1_data_compressed(data):
    fig, axs = plt.subplots(2,1, sharex=True)
    # Data and compressed
    axs[0].plot(data.self.x_data, data.y,
                color = colors['AuroraBlue'], lable='Original')
    axs[0].plot(data.xc, data.yc,
                '-o', color = colors['AuroraCyan'], label ='Compressed')
    axs[1].legend()
    # Residual relative to tolerance
    axs[1].plot(data.x, data.residuals_relative, color = colors['AuroraBlue'])
    axs[1].plot([data.x[0],data.x[-1]], [-1,-1], color = colors['AuroraCyan'])
    axs[1].plot([data.x[0],data.x[-1]], [1,1], color = colors['AuroraCyan'])
    axs[1].legend(['Relative Residual'])
    fig.tight_layout()
    plt.show()

class Debug:

    def __init__(self, x_data=None, y_data=None, ytol=2e-2, errorf='maxmaxabs', fitf='poly1'):
        self.x_data , self.y_data = ref.raw_sine(1e3) if y_data is None else (x_data, y_data)
        self.start = 1 # Index of starting point for looking for optimum
        self.end = len(self.x_data) - 2 # Number of uncompressed datapoints -2, i.e. the index
        self.offset = -1
        self.fit = None
        self.ytol = np.array(ytol)
        self.errorf = ls.errorfunctions[errorf]
        self.fitf = ls.fitfunctions[fitf]

        if len(self.y_data.shape) == 1: # Converting to correct shape for this function
            self.y_data = self.y_data.reshape(len(self.x_data),1)
        elif self.y_data.shape[0] != len(self.x_data):
            self.y_data = self.y_data.T
        
        
        self.x_c, self.y_c = [self.x_data[0]], [self.y_data[0]]
        if fitf == 'poly1' or 'poly2_2':
            self.y0 = [self.y_c[0]]
        elif fitf == 'poly3':
            self.y0 = [self.y_c[0],2*math.pi,0]
        # Solvers stuff
        self.fit1 = self.fit2 = None
        self.x1 = self.x_mid =  self.x2 = self.y1 = self.y_mid = self.y2 = None
        self.limit = self.estimate = None

        self.fig, self.axs = plt.subplots(3,1)
        for ax in self.axs:
            ax.grid()
        self.axs[2].set_ylabel('Tolerance left')
        self.axs[0].plot(self.x_data, self.y_data+self.ytol, color='blue')
        self.axs[0].plot(self.x_data, self.y_data-self.ytol, color='blue')

        self.line_fit, = self.axs[0].plot(0,0,'-',color='orange')
    #───────────────────────────────────────────────────────────────────
    def run(self):
        plt.ion()
        plt.show()
        self.x_compressed, self.y_compressed = self.LSQ10()
        plt.ioff()
        print(f'Compression factor {(1-len(self.x_compressed)/len(self.x_data))*100:.1f} %')
        pass
    #───────────────────────────────────────────────────────────────────
    def __getitem__(self, items):
        return (self.x_data[items], self.y_data[items])
    #───────────────────────────────────────────────────────────────────
    def LSQ10(self):
        '''Compresses the data of 1-dimensional system of equations
        i.e. single input variable and one or more output variable
        '''
        
        #───────────────────────────────────────────────────────────────
        def _f2zero(n):
            '''Function such that n is optimal when f2zero(n) = 0'''
            indices_sel = np.linspace(self.start, n + self.start, int((n+1)**0.5)+ 2).astype(int)
            indices_all = np.arange(self.start,self.start + int(n)+1)
            self.x_plot, y = self[indices_all]
            res_sel, y_next, a = self.fitf(self.x_data[indices_sel],
                                        self.y_data[indices_sel],
                                        self.x_c[-1], self.y0)
            Dx = self.x_plot.reshape([-1,1]) - self.x_c[-1]
            # y_fit = a*Dx*Dx*Dx + self.y0[2]*Dx*Dx + self.y0[1]*Dx + self.y0[0]
            # self.y_plot = a[0]*Dx*Dx + a[1]*Dx + self.y0[0]
            self.y_plot = a*Dx + self.y0[0]
            res_all = self.y_plot-y

            self.line_fit.set_xdata(self.x_plot)
            self.line_fit.set_ydata(self.y_plot)

            self.axs[1].clear()
            self.axs[1].grid()
            self.axs[1].set_ylabel('Residual relative to tolerance')
            self.axs[1].plot(indices_all-self.start, np.abs(res_all)/self.ytol -1,
                             '.', color = 'blue', label='ignored')
            self.axs[1].plot(indices_sel-self.start, np.abs(res_sel)/self.ytol-1,
                             'o', color='red', label='sampled')
            self.axs[1].legend()
            # Selected
            print(self)
            input('Fitting\n')
            if y_next is None:
                print('y_next is none')
            print(self.errorf(res_sel,self.ytol))
            return self.errorf(res_sel,self.ytol), y_next
        #───────────────────────────────────────────────────────────────

        for _ in range(int(len(self.x_data)/2)): # for to prevent infinite loop
            print(self)
            input('Next iteration\n')
            self.start += self.offset + 1 # self.start shifted by the number compressed
            # Estimated number of lines needed
            lines = ls.n_lines(self.x_data[self.start:],
                               self.y_data[self.start:],
                               self.x_c[-1], self.y_c[-1], self.ytol)
            # Arithmetic mean between previous step length and line self.estimate,
            # limited to self.end index of the array
            self.estimate = min(self.end, np.amin(((self.offset + (self.end+1) / lines)/2)).astype(int))
            err0 = abs(self.y_c[-1]-self.y_data[self.start-1])/self.ytol-1
            print(f'error? {err0}')
            self.x2 = self.estimate
            self.limit = self.end
            print(self)
            input('Ready to call droot\n')
            self.offset, self.fit = self.droot(_f2zero, err0*2)
            self.y0 = self.fit
            self.axs[2].clear()
            self.axs[2].grid()
            self.axs[2].set_ylabel('Maximum residual')
            self.axs[0].plot(self.x_plot, self.y_plot, color='red')
            self.end -= self.offset + 1

            if self.end > 0:
                self.x_c.append(self.x_data[self.offset + self.start])
                self.y_c.append(self.fit[0])
            else:
                # Last data point is same as in the uncompressed data
                self.x_c[-1] = self.x_data[-1]
                self.y_c[-1] = self.y_data[-1]
                break
        else:
            raise Warning('Maximum number of iterations reached, something is wrong')

        return np.array(self.x_c).reshape(-1,1), np.array(self.y_c)
    #───────────────────────────────────────────────────────────────────
    def __str__(self):
        s = ''
        s += f'Start\t\t{self.start}\n'
        s += f'End\t\t{self.end}\n'
        s += f'Offset\t\t{self.offset}\n'
        s += f'Y0\t\t{self.y0}\n'
        s += f'Fit\t\t{self.fit}\n'
        s += f'Fit1\t\t{self.fit1}\n'
        s += f'Fit2\t\t{self.fit2}\n'
        s += f'Estimate\t{self.estimate}\n'
        s += f'Limit\t\t{self.limit}\n'
        s += f'x1\t\t{self.x1}\n'
        s += f'x_mid\t\t{self.x_mid}\n'
        s += f'x2\t\t{self.x2}\n'
        s += f'y1\t\t{self.y1}\n'
        s += f'y_mid\t\t{self.y_mid}\n'
        s += f'y2\t\t{self.y2}'
        return s
    #───────────────────────────────────────────────────────────────────
    def interval(self, f):
        '''Returns the last x where f(x)<0'''
        mid, = self.axs[2].plot(self.x1, self.y1,'.', color='blue')
        while self.x2 - self.x1 > 2:
            print(self)
            input('Calculating new attempt in interval\n')
            # Arithmetic mean between linear estimate and half
            self.x_mid = int((self.x2 + self.x1)/2) + 1
            if self.x_mid == self.x1:    # To stop repetition in close cases
                print(self)
                input('Midpoint same x as lower one\n')
                self.x_mid += 1
            elif self.x_mid == self.x2:
                print(self)
                input('Midpoint same x as upper one\n')
                self.x_mid -= 1

            self.y_mid, fit = f(self.x_mid)
            mid.set_xdata(self.x_mid)
            mid.set_ydata(self.y_mid)
            if self.y_mid > 0:
                print(self)
                input('Error over tolerance\n')
                self.axs[2].plot(self.x2, self.y2,'.', color='black')
                self.x2, self.y2 = self.x_mid, self.y_mid
                self.xy2.set_xdata(self.x2)
                self.xy2.set_ydata(self.y2)
            else:
                print(self)
                input('Error under tolerance\n')
                self.axs[2].plot(self.x1, self.y1,'.', color='black')
                self.x1, self.y1, self.fit1 = self.x_mid, self.y_mid, fit
                self.xy1.set_xdata(self.x1)
                self.xy1.set_ydata(self.y1)

        if self.x2 - self.x1 == 2: # Points have only one point in between
            print(self)
            input('Points have only one point in between\n')
            self.y_mid, fit = f(self.x1+1) # Testing that point
            return (self.x1+1, fit) if (self.y_mid <0) else (self.x1, self.fit1) # If under, give that fit
        else:
            print(self)
            input('Points have no point in between\n')
            return self.x1, self.fit1
    #───────────────────────────────────────────────────────────────────
    def droot(self, f, y0):
        '''Finds the upper self.limit to interval
        '''
        self.x1, self.y1 = 0, y0
        self.xy1,  = self.axs[2].plot(self.x1, self.y1,'.', color='green')

        self.y2, self.fit2 = f(self.x2)
        self.xy2, = self.axs[2].plot(self.x2, self.y2,'.', color='blue')
        self.fit1 = None
        while self.y2 < 0:
            print(self)
            input('Calculating new attempt in droot\n')
            self.axs[2].plot(self.x1, self.y1,'.', color='black')
            self.x1, self.y1, self.fit1 = self.x2, self.y2, self.fit2
            self.xy1.set_xdata(self.x1)
            self.xy1.set_ydata(self.y1)
            self.x2 *= 2
            self.xy2.set_xdata(self.x2)
            if self.x2 >= self.limit:
                self.axs[2].plot([self.limit,self.limit], [self.y1,0],'.', color='blue')
                self.y2, self.fit2 = f(self.limit)
                if self.y2<0:
                    print(self)
                    input('End reached within tolerance\n')
                    return self.limit, self.fit2
                else:
                    print(self)
                    input('End reached outside tolerance\n')
                    self.x2 = self.limit
                    break
            self.y2, self.fit2 = f(self.x2)
            self.xy2.set_ydata(self.y2)
        self.xy2.set_color('red')
        print(self)
        input('Points for interval found\n')
        return self.interval(f)