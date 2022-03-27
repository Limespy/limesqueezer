'''
Examples
========================================================================

A series of examples on how to use this package
'''
import API as ls
import reference

# You have some data from system of equations
# For example, let's take 100 000 datapoints along some 2nd degree polynomial
example_x_array, example_y_array = reference.raw['poly2'](n=1e5)
# Now you want to compress it with maximum error being 
example_x_array_compressed, example_y_array_compressed, _ = ls.compress(example_x_array, example_y_array)

# Or maybe you have some generator-like thing that gives out numbers.
# E.g. PBOS simulation step
# Here you use the context manager "Stream"
# Initialise with first values, here I am just going to use the first
example_x0, example_y0 = example_x_array[0], example_y_array[0]
example_generator = zip(example_x_array[1:], example_y_array[1:])

with ls.Stream(example_x0, example_y0) as compressed:
    for example_x, example_y in example_generator:
        compressed(example_x, example_y)
print('And now you have your datastream stored in compressed format')
print(compressed.x)
print(compressed.y)

print(compressed.x.shape)
print(example_x_array_compressed.shape)