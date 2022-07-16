#!/usr/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import limesqueezer as ls
import numpy as np

def data_compressed_1d(input_x, input_y, tolerance):
    
    output_x, output_y = ls.compress(input_x, input_y, tolerances = tolerance)

    function = ls.decompress(output_x, output_y)

    fig, axs = plt.subplots(2,1, sharex=True)

    
    # Data and compressed
    axs[0].plot(input_x, input_y, lable='Original')
    axs[0].plot(output_x, output_y, '-o', label ='Compressed')
    axs[0].legend()
    
    # Residuals to tolerance
    axs[1].plot(input_x, function(input_x) - input_y, label = 'Residuals')
    axs[1].hlines(y = tolerance)
    axs[1].legend(['Relative Residual'])
    
    fig.tight_layout()
    plt.show()
#───────────────────────────────────────────────────────────────────────
def comparison(x: np.ndarray, y1: np.ndarray, y2: np.ndarray):
    xlen = len(x)
    if len(y1.shape) == 1:
        shape = (1, xlen)
    elif y1.shape[0] == xlen:
        shape = (y1.shape[1], xlen)
    else:
        shape = (y1.shape[0], xlen)

    y1 = ls.to_ndarray(y1, shape)
    y2 = ls.to_ndarray(y2, shape)
    fig, axs = plt.subplots(2, shape[0], sharex = True, squeeze = False)
    for y1n, y2n, ax in zip(y1, y2, axs[0]):
        # Data and compressed
        ax.plot(x, y1n, linewidth = 5, label = '1')
        ax.plot(x, y2n, '.', linewidth = 1, label = '2')
        ax.legend()
    for y1n, y2n, ax in zip(y1, y2, axs[1]):
        # Residuals to tolerance
        ax.plot(x, y2n - y1n, label = 'Residuals')
        ax.legend()

    fig.tight_layout()
    plt.show()
#───────────────────────────────────────────────────────────────────────
def simple(x, y):
    shape = (len(x), -1)
    y = ls.to_ndarray(y, shape)
    plt.plot(x, y)
    plt.show()

