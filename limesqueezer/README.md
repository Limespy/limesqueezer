# interp-compression
Lossy compression of numerical data using interpolation

- [interp-compression](#interp-compression)
  - [Use](#use)
  - [Compression methods](#compression-methods)
    - [Split](#split)
    - [Pick](#pick)
    - [Interp10](#interp10)
  - [Input methods](#input-methods)
    - [Stream](#stream)
    - [Block](#block)
  - [Output types](#output-types)
    - [Monolith](#monolith)
    - [Pairs](#pairs)
    - [Both](#both)
    - [Smaller](#smaller)
  - [Math stuff](#math-stuff)

## Use

- Input as stream or block?
- Compression method?
- Output type?

## Compression methods

### Split

Fastest method, but lowest compression ratio. 
Outputs only the indices of the data to be selected

### Pick

Outputs only the indices of the data to be selected

### Interp10



## Input methods

### Stream

Context manager and a class.

- Data is fed one point at the time.
- Context manager is used to ensure proper finishing of the compression process.

### Block

A function.

The whole of data is given as input.

## Output types

Default is monolith as it usually gives better total compression

### Monolith

tuple of two numpy ndarrays (x, y)

### Pairs

Pairs of (x, y_n) as a list containing tuples containing numpy ndarrays of x and corresponding y


[(x1, y1), (x2, y2), (x3, y3), ..., (xn, yn)]



### Both
_Supplementary feature_


Tuple containing both monolith and split output

### Smaller
_Supplementary feature_

Computes both monolith and split output and outputs the smaller of the two



## Math stuff

$$
L(x) = \frac{y_2-y_1}{x_2 - x_1} \cdot (x - x_1) + y_1
$$

$$
x_n = \frac{x(n) + x(n+1)}{2}
$$

$$
y_n = \frac{y(n) + y(n+1)}{2}
$$

$$
L(x) = \frac{y_n-y_0}{x_n - x_0} \cdot (x - x_0) + y_0
$$

$$
L(x,n) = \frac{y_n(n)-y_0}{x_n(n) - x_0} \cdot (x - x_0) + y_0
$$

$$
err(n) = L(x_{data},n) - y_{data}
$$

$$
P_2(x) = a \cdot x^2 + b \cdot x + c
$$

$$
\dot{P}_2(x) = 2 \cdot a \cdot x + b
$$
System of equations

$$
\dot{P}_2(x) = 2 \cdot a \cdot x_0 + b
$$
$$
P_2(x_0) = a \cdot x_0^2 + b \cdot x_0 + c = y_0
$$
$$
P_2(x_1) = a \cdot x_1^2 + b \cdot x_1 + c = y_1
$$


Given atol and Delta_y, 
in the best case 1 line would be enough 
and in the worst case Delta_y / atol.

Geometric mean between these would maybe be good choice,
so likely around n_lines ~ sqrt(Delta_y / atol)
meaning delta_x ~ Delta_x * sqrt(atol / Delta_y)

When this is normalised so Delta_y = 1 and Delta_x = n,
delta_x ~ n * sqrt(atol)