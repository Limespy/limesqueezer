#!/usr/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

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