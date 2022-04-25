'''
Examples
========================================================================

A series of examples on how to use this package
'''
#%%═════════════════════════════════════════════════════════════════════
# IMPORT
# limesqueezer uses numpy ndarrays types for input and output
import numpy as np 
# The package itself with author-recommended abbreviation.
# Rest of documentation uses this abbreviation.
import limesqueezer as  ls 

# You have some data from system of equations
# For example, let's take 100 000 datapoints along some function
input_x = np.linspace(0,1,int(1e5))
input_y = np.sin(6 * input_x * input_x)
# Now you want to compress it with maximum absolute error being 1e-3.
tolerance = 1e-3

# Or maybe you have some generator-like thing that gives out numbers.
# E.g. some simulation step
# Here you use the context manager "Stream"
# Initialise with first values, here I am just going to use the first

#%%═════════════════════════════════════════════════════════════════════
# BLOCK

# To simplify the interface, the package has beem made callable.
output_x, output_y = ls(input_x, input_y)


# You can also use 
output_x, output_y = ls.compress(input_x, input_y)
# that is more more comfortable to use.

#%%═════════════════════════════════════════════════════════════════════
# STREAM

example_x0, example_y0 = input_x[0], input_y[0]
generator = zip(input_x[1:], input_y[1:])

# The context manager for Stream data is 'Stream'.

with ls.Stream(example_x0, example_y0) as record:
    # A side mote: In Enlish language the word 'record' can be either
    # verb or noun and since it performs this double role of both taking
    # in data and being storage of the data, it is a fitting name for the object

    # For sake of demonstarion
    print(record.state)
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

# 

output_x, output_y = record.x, record.y
print(record.state)
print(record.__str__)