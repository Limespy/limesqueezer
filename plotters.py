#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pathlib

import limesqueezer as ls
import matplotlib.pyplot as plt
import numpy as np
#───────────────────────────────────────────────────────────────────────
def data_compressed_decompressed_1d(x_data,
                                    y_data,
                                    x_compressed,
                                    y_compressed,
                                    y_decompressed,
                                    tolerances,
                                    fname,
                                    is_show = False
                                    ):

    fig, axs = plt.subplots(2,1, sharex=True)
    # Data and compressed
    axs[0].plot(x_data, y_data, label='Original')
    axs[0].plot(x_compressed, y_compressed, '-o', label ='Compressed')
    axs[0].legend()

    # Residuals to tolerance
    residuals = y_decompressed - y_data
    total_tolerance = ls._tolerancefunctions[0](y_data, tolerances)

    axs[1].plot(x_data, residuals, label = 'Residuals')
    axs[1].plot(total_tolerance, label = 'Total tolerance', color = 'red')
    axs[1].plot(-total_tolerance, color = 'red')

    axs[1].legend()

    fig.tight_layout()
    # Instead of showing the figure it is saved as png
    if fname:
        plt.savefig(pathlib.Path(__file__).parent / 'figures' / fname,
                    bbox_inches = 'tight')
    if is_show:
        plt.show()
#───────────────────────────────────────────────────────────────────────
def plot_tolerances(x_data, y_data, tolerances):
    x_compressed, y_compressed = ls.compress(x_data,
                                             y_data,
                                             tolerances = tolerances)
    function = ls.decompress(x_compressed, y_compressed)
    y_decompressed = function(x_data).reshape(y_data.shape)


#───────────────────────────────────────────────────────────────────────
def comparison(x: ls.FloatArray,
              y1: ls.FloatArray,
              y2: ls.FloatArray):
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
