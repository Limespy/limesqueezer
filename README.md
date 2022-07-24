# Overview

[![PyPI Package latest release](https://img.shields.io/pypi/v/limesqueezer.svg)](https://pypi.org/project/limesqueezer)
[![PyPI Wheel](https://img.shields.io/pypi/wheel/limesqueezer.svg)](https://pypi.org/project/limesqueezer)
[![Supported versions](https://img.shields.io/pypi/pyversions/limesqueezer.svg)](https://pypi.org/project/limesqueezer)
[![Supported implementations](https://img.shields.io/pypi/implementation/limesqueezer.svg)](https://pypi.org/project/limesqueezer)
[![Commits since latest release](https://img.shields.io/github/commits-since/Limespy/limesqueezer/v1.0.12.svg)](https://github.com/limespy/limesqueezer/compare/v1.0.11...master)

Lossy compression with controlled error tolerance for smooth data series


- [Overview](#overview)
- [Use](#use)
  - [Compression](#compression)
    - [Common parameters](#common-parameters)
      - [Tolerances](#tolerances)
    - [Block](#block)
    - [Stream](#stream)
  - [Decompression](#decompression)
  - [Combining compression methods](#combining-compression-methods)
- [Meta](#meta)
  - [Version numbering](#version-numbering)
  - [Changelog](#changelog)
    - [1.0.12 2022-07-16](#1012-2022-07-16)
    - [1.0.11 2022-07-16](#1011-2022-07-16)
    - [1.0.10 2022-05-08](#1010-2022-05-08)
    - [1.0.9 2022-04-03](#109-2022-04-03)
    - [1.0.8 2022-03-20](#108-2022-03-20)
    - [1.0.3 2021-11-30](#103-2021-11-30)

# Use

limesqueezer uses numpy ndarrays types for input and output.
package import name is `limesqueezer`.
Author recommend abbreviation `ls`
Rest of documentation uses this abbreviation.

``` python
    import numpy as np
    import limesqueezer as  ls
```

## Compression

### Common parameters

#### Tolerances

Keyword `tolerances`

Tolerances
Absolute Tolerance, Relative Tolerance and Falloff to smooth between them.


tolerances, Falloff determines how much the absolute error is
        reduced as y value grows.
            If 3 values: (relative, absolute, falloff)
            If 1 values: (relative, absolute, 0)
            If 1 value:  (0, absolute, 0)

Allowed deviation is calculated with following function

$$
deviation = Relative \cdot Y_{data} + \frac{Absolute}{Falloff \cdot Y_{data} + 1}
$$

$$
D_Y^1 deviation = Relative - \frac{Absolute \cdot Falloff}{(Falloff \cdot Y_{data} + 1)^2}
$$

To have constrain that

$$
D_Y^1 deviation(0) > 0 
$$
Means
$$
Relative > Absolute \cdot Falloff 
$$

Recommended

`errorfunction`


You have some data from system of equations
For this example, let's make 100 000 datapoints along some function
``` python
    input_x = np.linspace(0, 1, int(1e4))
    input_y = np.sin(24 * input_x ** 2)
```
Example of the data, compression output, and residuals
![Example of the data, compression output, and residuals](figures/example.png)

Or maybe you have some generator-like thing that gives out numbers.
E.g. some simulation step
Here you use the context manager "Stream"
Initialise with first values, here I am just going to use the first

### Block

A function.

The whole of data is given as input.

To simplify the interface, the package has beem made callable.
Now you want to compress it with maximum absolute error being 1e-3.

``` python
    output_x, output_y = ls(input_x, input_y, tol = 1e-3)
```

You can also use

``` python
    output_x, output_y = ls.compress(input_x, input_y, tol = 1e-3)
```
if that is more comfortable for you.

### Stream

Context manager and a class.

- Data is fed one point at the time.
- Context manager is used to ensure proper finishing of the compression process.

``` python
    example_x0, example_y0 = input_x[0], input_y[0]
    generator = zip(input_x[1:], input_y[1:])
```
The context manager for Stream data is 'Stream'.

``` python
    with ls.Stream(example_x0, example_y0, tol = 1e-3) as record:
        for example_x_value, example_y_value in generator:
            record(example_x_value, example_y_value)
```
Using record.x or record.y in the with statement block results in
attribute error, as those attributes are generated only when 
the record is closed.

If you want to access the data fed to the record, you can use
``` python
    x_compressed, y_compressed = record.xc, record.yc
```
to access the already compressed data and

``` python
    x_buffered, y_buffered = record.xb, record.yb
```
to access the buffered data waiting more values or closing of
the record to be compressed.



``` python
    output_x, output_y = record.x, record.y
    print(record.state)
    print(record)
```

A side mote: In English language the word 'record' can be either
verb or noun and since it performs this double role of both taking
in data and being storage of the data, it is a fitting name for the object





## Decompression

Decompression is done in two main steps with interpolation.
First an interpolation function is created
Then that is called.

This two-step approach allows more flexible use of the data.

``` python

```

## Combining compression methods

This compression method can be combined with lossless compressiom to achieve even higher compression ratios.
The lossless compression should be done only after the lossy compression this package provides.


# Meta

## Version numbering

Version code is composed of three numbers:
Major, Minor, Micro

Experimental, alpha or beta versions are indicated by a 0 as one of those three.

First public release starts with Major Version.
Incrementation of Major Version indicates backwards compatibility breaking change in API or fuctionality.

Minor Version indicates design 

While the Minor Version is 0, the package is in alpha stage. That means features and API

Later incrementation of the Minor Version signifies upgrades to the features and interfaces.
In general changes here mean changes in the design and specification, but not such that it breaks backwards compatibility
I.e. code that works with _documented_ features of X.n.x will work with X.n+1.y

Micro Version indicate implementation.
These are bugfixes, typo corrections, documentation clarifications.
In Micro Version incementation the _intention_, i.e. intended documented specification of the Minor Version is not changed,
only the implementation.

## Changelog

### 1.0.12 2022-07-16

- Changed README to Markdown-only

### 1.0.11 2022-07-16

- Debug plotting improvements
- Added undocumented API for other fitting functions
- More tests
- Profiling and benchmarking from tests
- 

### 1.0.10 2022-05-08


- Cleaned documentation

### 1.0.9 2022-04-03


- Block and stream compression are much more uniform
- Restructuring
- Tests
- Profiling

### 1.0.8 2022-03-20


- Step-by-step style ploting of the compression.

### 1.0.3 2021-11-30


- First release on PyPI.


