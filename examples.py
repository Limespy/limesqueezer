'''
Examples
========================================================================

A series of examples on how to use this package
'''
#%%═════════════════════════════════════════════════════════════════════
# IMPORT

# limesqueezer uses numpy ndarrays types for input and output
import numpy as np 

import matplotlib.pyplot as plt
import pathlib
# The package itself with author-recommended abbreviation.
# Rest of documentation uses this abbreviation.
import limesqueezer as  ls 

# You have some data from system of equations
# For example, let's take 100 000 datapoints along some function
input_x = np.linspace(0,1,int(1e4))
input_y = np.sin(24 * input_x ** 2)
# Now you want to compress it with maximum absolute error being 1e-3.
tolerance = (1e-2, 1e-3, 1)

# Or maybe you have some generator-like thing that gives out numbers.
# E.g. some simulation step
# Here you use the context manager "Stream"
# Initialise with first values, here I am just going to use the first

#%%═════════════════════════════════════════════════════════════════════
# BLOCK

# To compress a block simply
output_x, output_y = ls.compress(input_x, input_y, tolerances = tolerance, errorfunction = 'maxRMS_absend')
# that is more more comfortable to use.

#%%═════════════════════════════════════════════════════════════════════
# STREAM

example_x0, example_y0 = input_x[0], input_y[0]
generator = zip(input_x[1:], input_y[1:])

# The context manager for Stream data is 'Stream'.

with ls.Stream(example_x0, example_y0, tolerances = tolerance, errorfunction = 'maxRMS_absend') as record:
    # A side mote: In Enlish language the word 'record' can be either
    # verb or noun and since it performs this double role of both taking
    # in data and being storage of the data, it is a fitting name for the object

    # For sake of demonstarion
    print(f'Record state within the context manager is {record.state}')
    for example_x_value, example_y_value in generator:
        record(example_x_value, example_y_value)

    # Using record.x or record.y in the with statement block results in
    # attribute error, as those attributes are generated only when 
    # the record is closed.
    # If you want to access the data fed to the record, you can use
    x_compressed, y_compressed = record.xc, record.yc
    # to access the already compressed data and
    x_buffered, y_buffered = record.xb, record.yb
    # to access the buffered data waiting more values or closing of
    # the record to be compressed.

output_x, output_y = record.x, record.y
print(f'Record state after the context manager is {record.state}' )

function = ls.decompress(output_x, output_y)
decompressed_y = function(input_x).flatten()

fig, axs = plt.subplots(2,1, sharex=True)

# Data and compressed
axs[0].plot(input_x, input_y, label='Original')
axs[0].plot(output_x, output_y, '-o', label ='Compressed')
axs[0].legend()

# Residuals to tolerance
axs[1].plot(input_x, decompressed_y - input_y, label = 'Residuals')
axs[1].hlines(y = tolerance, xmin = input_x[0], xmax = input_x[-1], color = 'red', label = 'Tolerance')
axs[1].hlines(y = -tolerance, xmin = input_x[0], xmax = input_x[-1], color = 'red')
axs[1].legend()

fig.tight_layout()
plt.savefig(pathlib.Path(__file__).parent / 'figures' / 'example.png', bbox_inches = 'tight')