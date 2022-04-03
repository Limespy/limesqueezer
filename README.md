# interp-compression

Lossy compression tools for smooth data series.
WIP

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
      - [Sqrt heuristic 1](#sqrt-heuristic-1)
      - [Sqrt heuristic 2](#sqrt-heuristic-2)
      - [Sqrt heuristic 3](#sqrt-heuristic-3)
  - [Visualiser](#visualiser)
    - [Plot](#plot)
      - [Top](#top)
      - [Mid](#mid)
      - [Bottom](#bottom)

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

#### Sqrt heuristic 1

$$
interp(x) = a \cdot \sqrt{x} + b
$$
$$
interp(x_1) = y_1 = a \cdot \sqrt{x_1} + b
$$
$$
interp(x_2) = y_2 = a \cdot \sqrt{x_2} + b
$$

So
$$
y_2 - y_1 = a \cdot (\sqrt{x_2} - \sqrt{x_1})
$$
$$
a = \frac{y_2 - y_1}{\sqrt{x_2} - \sqrt{x_1}}
$$
And
$$
b = y_1 - a \cdot \sqrt{x_1}
$$

Combined back
$$ 
interp(x) = a \cdot \sqrt{x} + y_1 - a \cdot \sqrt{x_1}
 = \frac{y_2 - y_1}{\sqrt{x_2} - \sqrt{x_1}} \cdot (\sqrt{x} - \sqrt{x_1}) + y_1
$$
And when wanting
$$
interp(x_0) = 0 = \frac{y_2 - y_1}{\sqrt{x_2} - \sqrt{x_1}} \cdot (\sqrt{x_0} - \sqrt{x_1}) + y_1
$$
$$
\sqrt{x_1} - y_1 \cdot \frac{\sqrt{x_2} - \sqrt{x_1}}{y_2 - y_1} = \sqrt{x_0}
$$

$$
x_0 = (\sqrt{x_1} - y_1 \cdot \frac{\sqrt{x_2} - \sqrt{x_1}}{y_2 - y_1})^2
$$

#### Sqrt heuristic 2

$$
interp(x) = a \cdot \sqrt{x - x_1} + y_1
$$
$$
interp(x_2) = y_2 = a \cdot \sqrt{x_2 - x_1} + y_1
$$

$$
a = \frac{y_2 - y_1}{\sqrt{x_2 - x_1}}
$$

Combined
$$
interp(x) = \frac{y_2 - y_1}{\sqrt{x_2 - x_1}} \cdot \sqrt{x - x_1} + y_1
$$

$$
interp(x_0) = 0 = \frac{y_2 - y_1}{\sqrt{x_2 - x_1}} \cdot \sqrt{x_0 - x_1} + y_1
$$

$$
x_0 = x_1 + (x_2 - x_1) \cdot (\frac{y_1}{y_2 - y_1})^2
$$

#### Sqrt heuristic 3

$$ 
interp(x) = a \cdot \sqrt{x - b} + c
$$
Where
$$ 
interp(0) = 0 = a \cdot \sqrt{b} + c
$$
$$ 
interp(x_1) = y_1 = a \cdot \sqrt{x_1 - b} + c
$$
$$ 
interp(x_2) = y_2 = a \cdot \sqrt{x_2 - b} + c
$$
Combining some
$$ 
y_1 = a \cdot (\sqrt{x_1 - b} - \sqrt{b})
$$
$$ 
y_2 = a \cdot (\sqrt{x_2 - b} - \sqrt{b})
$$

$$ 
\frac{y_2}{y_1} = \frac{(\sqrt{x_2 - b} - \sqrt{b})}{(\sqrt{x_1 - b} - \sqrt{b})}
$$
$$ 
y_2 \cdot (\sqrt{x_1 - b} - \sqrt{b}) = y_1 \cdot (\sqrt{x_2 - b} - \sqrt{b})
$$

$$ 
y_2 \cdot \sqrt{x_1 - b} - y_2 \cdot \sqrt{b} = y_1 \cdot \sqrt{x_2 - b} - y_1 \cdot\sqrt{b}
$$
$$ 
y_2 \cdot \sqrt{x_1 - b} + (y_1 - y_2) \cdot \sqrt{b} - y_1 \cdot \sqrt{x_2 - b} = 0
$$

## Visualiser

Three domains --> three subplots
1. Data and fitted points in (x, y)
2. Of the latest fit in (x, error)
3. Reference error vesus the reach of the fit (n, toleranced error)


Loop is:
1. Calculate some step
2. Update plots
3. Wait for user input to continue

### Plot


#### Top

1. Given data
2. Compressed points with lines
3. Next point candidate with line to last compressed


#### Mid

1. Residuals
2. Residuals of candidate point and line


#### Bottom

1. Previaous attempts
2. Left and right
3. 
