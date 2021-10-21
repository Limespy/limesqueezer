#!/usr/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

colors = {'AuroraBlue': '#0e1c3f',
          'AuroraCyan': '#00a0c3'}

def plot_1_data_compressed(data):
    fig, axs = plt.subplots(2,1, sharex=True)
    # Data and compressed
    axs[0].plot(data.x, data.y, color = colors['AuroraBlue'] )
    axs[0].plot(data.xc, data.yc, '-o', color = colors['AuroraCyan'] )
    axs[0].legend(['Original', 'Compressed'])
    # Residual relative to tolerance
    axs[1].plot(data.x, data.residuals_relative, color = colors['AuroraBlue'])
    fig.tight_layout()
    plt.show()

