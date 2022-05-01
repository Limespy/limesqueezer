#!/usr/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import limesqueezer as ls
import numpy as np

def plot_1_data_compressed(input_x, input_y, tolerance):
    
    output_x, output_y = ls.compress(input_x, input_y, tol = tolerance)

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
