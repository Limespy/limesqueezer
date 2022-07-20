#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Limesqueezer
============

Lossy compression tools for smooth data series.

Compression factors from 2:1 to 100:1 are achived,
but do depend heavily on the input data and tolerance.

Main function are ``compress`` and ``decompress``.

As you probably expected, ``compress`` takes in arrays of data to be compressed and outputs compressed data arrays

``decompress`` takes in data arrays to be decompressed and returns interpolation function.
With this function you can then estimate any values within the range of the input data.
Since this operates on data arrays, you can use ``decompress`` for interpolation without using the ``compress`` function in first place. 

Compression is done by fitting some function(s) to the data.

There is be multiple builtin compressor/interpolators and a compressor/interpolator generator for more general


Use and examples
================

Author recommends abbreviation ``ls``

Like so::

    import limesqueezer as ls

For these examples let's set up mock data of 100 000 data points::

    input_x = np.linspace(0, 6, 100_000)
    input_y = np.sin(input_x ** 2)

Compressing a block of data
---------------------------

Now we have data to compress, so let's start with just the defaults.

    import limesqueezer as ls
    x_compressed, y_compressed = ls.compress(input_x, input_y)

Simple as that!

With lossy compression tolerance has critical role and controlling it is one of the main motivations behind this package.
Currently ``compress`` has keyword argument for absolute tolerance ``tol``.
Unfortunately relative tolerance has not yet been implemented.

Let's use arbitrary tolerance of 0.001::

    x_compressed, y_compressed = ls.compress(input_x, input_y,
                                             tolerances = (1e-3, 1e-4, 1))

Similarly the compressor can be selected with a keyword argument::

    x_compressed, y_compressed = ls.compress(input_x, input_y,
                                             tolerances = (1e-3, 1e-4, 1),
                                             compressor = 'Poly10')

Compressor keyword accepts a custom compressor function.

    x_compressed, y_compressed = ls.compress(input_x, input_y, 
                                             tolerances = (1e-3, 1e-4, 1),
                                             compressor = custom_compressor)

The custom function must have following interface::

    function(x: np.ndarray, y: np.ndarray, x0: float, y0: np.ndarray)
        -> residuals: np.ndarray, y1: np.ndarray,

Where:
|    name   | description                          | shape |
| :-------: | ------------------------------------ | :---: |
|      x    | x values of the points to be fitted  |  (n,)
|      y    | y values of the points to be fitted  | (n,m)
|     y0    | Last compressed point y value(s)     | (1, m)
| residuals | Residuals of the fit, i.e. y_fit - y | (n, m)
|     y1    | Next Y values                        | (1, m)


=========  ====================================  =======
name       description                           shape
=========  ====================================  =======
x          x values of the points to be fitted    (n,)
y          y values of the points to be fitted   (n,m)
y0         Last compressed point y value(s)      (1, m)
residuals  Residuals of the fit, i.e. y_fit - y  (n, m)
y1         Next Y values (1, m)
=========  ====================================  =======

Compressing a data stream
------------------------------------

Decompressing
-------------

For decom

For this reason, the decompressor can in

Making custom compressors
==================



A custom compressor needs a fucntion

'''
import sys
__version__ = '1.0.12'
#%%═════════════════════════════════════════════════════════════════════
# IMPORT
from .API import *
# A part of the hack to make the package callable
sys.modules[__name__].__class__ = Pseudomodule