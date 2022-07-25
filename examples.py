'''
Examples
========================================================================

A series of examples on how to use this package
'''
#%%═════════════════════════════════════════════════════════════════════
# QUICK START BLOCK

import numpy as np 

import limesqueezer as  ls 

x_data = np.linspace(0, 1, int(1e4))
y_data = np.sin(24 * x_data ** 2)

tolerance = 0.05
x_compressed, y_compressed = ls.compress(x_data, y_data, tolerances = tolerance)

x0, y0 = x_data[0], y_data[0]
generator = zip(x_data[1:], y_data[1:])

with ls.Stream(x0, y0, tolerances = 0.01) as record:
    for x_value, y_value in generator:
        record(x_value, y_value)

x_compressed, y_compressed = record.x, record.y

# Decompression
function = ls.decompress(x_compressed, y_compressed)
y_decompressed = function(x_data).reshape(y_data.shape)

print(y_decompressed)

residuals = y_decompressed - y_data
maximum_error = np.amax(np.abs(residuals))
print(f'Maximum error should be ~= {tolerance}: {maximum_error:.5f}')

from matplotlib import pyplot as plt

fig, axs = plt.subplots(2,1, sharex=True)
# Data and compressed
axs[0].plot(x_data, y_data, label='Original')
axs[0].plot(x_compressed, y_compressed, '-o', label ='Compressed')
axs[0].legend()

# Residuals to tolerance
residuals = y_decompressed - y_data
axs[1].plot(x_data, y_decompressed - y_data, label = 'Residuals')
axs[1].axhline(tolerance, label = 'Total tolerance', color = 'red')
axs[1].axhline(-tolerance, color = 'red')
axs[1].legend()

fig.tight_layout()
# Instead of showing the figure it is saved as png
import pathlib
plt.savefig(pathlib.Path(__file__).parent / 'figures' / 'quick_start.png', bbox_inches = 'tight')

#%%═════════════════════════════════════════════════════════════════════

# #%%═════════════════════════════════════════════════════════════════════
# # STREAM

# example_x0, example_y0 = X_DATA[0], Y_DATA[0]
# generator = zip(X_DATA[1:], Y_DATA[1:])

# # The context manager for Stream data is 'Stream'.

# with ls.Stream(example_x0, example_y0, tolerances = tolerances, errorfunction = 'maxmaxabs') as record:
#     # A side mote: In Enlish language the word 'record' can be either
#     # verb or noun and since it performs this double role of both taking
#     # in data and being storage of the data, it is a fitting name for the object

#     # For sake of demonstarion
#     print(f'Record state within the context manager is {record.state}')
#     for example_x_value, example_y_value in generator:
#         record(example_x_value, example_y_value)

#     # Using record.x or record.y in the with statement block results in
#     # attribute error, as those attributes are generated only when 
#     # the record is closed.
#     #
#     # If you want to access the already compressed data
#     x_compressed, y_compressed = record.xc, record.yc
#     # and to access the buffered data waiting more values or closing of
#     # the record to be compressed.
#     x_buffered, y_buffered = record.xb, record.yb

# x_compressed, y_compressed = record.x, record.y
# print(f'Record state after the context manager is {record.state}' )

# # Side Note:
# # There is a third state called 'closing'.
# # In case the record closing process crashes somehow, the state is left to this.

# #%%═════════════════════════════════════════════════════════════════════
# # DECOMPRESSION

# function = ls.decompress(x_compressed, y_compressed)
# y_decompressed = function(X_DATA).flatten()


# #%%═════════════════════════════════════════════════════════════════════
# # Plotting

# def tolerance(y, tolerances):
#     return np.abs(y) * tolerances[0] + tolerances[1] / (np.abs(y) * tolerances[2] + 1)
# tolerance_total = tolerance(Y_DATA, tolerances)
# def plot():
#     import matplotlib.pyplot as plt
#     import pathlib
#     fig, axs = plt.subplots(2,1, sharex=True)
#     # Data and compressed
#     axs[0].plot(X_DATA, Y_DATA, label='Original')
#     axs[0].plot(x_compressed, y_compressed, '-o', label ='Compressed')
#     axs[0].legend()

#     # Residuals to tolerance
#     axs[1].plot(X_DATA, y_decompressed - Y_DATA, label = 'Residuals')
#     axs[1].plot(X_DATA, tolerance_total, label = 'Total tolerance', color = 'red')
#     axs[1].plot(X_DATA, -tolerance_total, color = 'red')
#     axs[1].legend()

#     fig.tight_layout()
#     plt.savefig(pathlib.Path(__file__).parent / 'figures' / 'example.png', bbox_inches = 'tight')