- [Math stuff](#math-stuff)
    - [Sqrt heuristic 1](#sqrt-heuristic-1)
    - [Sqrt heuristic 2](#sqrt-heuristic-2)
    - [Sqrt heuristic 3](#sqrt-heuristic-3)
- [Fit models](#fit-models)
  - [Polynomials](#polynomials)
- [Visualiser](#visualiser)
  - [Plot](#plot)
    - [Top](#top)
    - [Mid](#mid)
    - [Bottom](#bottom)
- [Publishing pipeline](#publishing-pipeline)
- [Tolerance](#tolerance)

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

## Fit models

### Polynomials

$$
P(x)_n = \sum_{e = 0}^{n} p_n \cdot x^e
$$

Where p_n is nth polynomial coefficient

For $3^{rd}$ degree polynomial

$$
P(x)
 = p_3 \cdot x_0^3
  + p_2 \cdot x^2
  + p_1 \cdot x + p_0
$$

$$
D_x^1(P)(x) = P'(x)
 = 3 \cdot p_3 \cdot x^2
  + 2 \cdot p_2 \cdot x
  + p_1
$$

$$
y_0 = P(0) = p_0
$$
$$
y_0' = P'(0) = p_1
$$

$$
y = P(\Delta x)
 = p_3 \cdot \Delta x^3
  + p_2 \cdot \Delta x^2
  + y_0' \cdot \Delta x + y_0
$$

$$
y' = P'(\Delta x)
 = 3 \cdot p_3 \cdot \Delta x^2
  + 2 \cdot p_2 \cdot \Delta x
  + y_0'
$$
So
$$
p_3 \cdot \Delta x^3 + p_2 \cdot \Delta x^2
 = y - y_0' \cdot \Delta x - y_0
$$
paramaters from two points
$$
y_1 = p_3 \cdot 0^3 + p_2 \cdot 0 ^2 + p_1 \cdot 0 + p_0
$$
$$
y_2 = p_3 \cdot \Delta x^3 + p_2 \cdot \Delta x ^2 + p_1 \cdot \Delta x + p_0
$$
$$
y_1' = 3 \cdot p_3 \cdot 0^2 + 2 \cdot p_2 \cdot 0+ p_1
$$
$$
y_2' = 3 \cdot p_3 \cdot \Delta x^2 + 2 \cdot p_2 \cdot \Delta x + p_1
$$

$$
y_1 = p_0
$$
$$
\Delta y = y_2 - y_1 = p_3 \cdot \Delta x^3 + p_2 \cdot \Delta x ^2 + p_1 \cdot \Delta x
$$
$$
y_1' = p_1
$$
$$
\Delta y' = y_2' - y_1' = 3 \cdot p_3 \cdot \Delta x^2 + 2 \cdot p_2 \cdot \Delta x
$$

$$
    \begin{array}{c}
    \Delta y \\
    \Delta y'
    \end{array}
  -
    \begin{array}{c}
    y_1 \cdot \Delta x \\
    0
    \end{array}
  =
    \begin{array}{cc}
    \Delta x^3 & \Delta x ^2\\
    3 \cdot \Delta x^2 & 2 \cdot \Delta x
    \end{array}
  \cdot
    \begin{array}{c}
    p_3 \\
    p_2
    \end{array}
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

1. Previous attempts
2. Left and right
3.


## Publishing pipeline

1. Running unit tests
2. Running integration and output consistency tests
3. Running examples and generating example plots
4. Running benchmarks
5. Profiling with and without numba
6. Converting README

## Tolerance

Later tolerance will be implemented as smooth function of $absolute$ and $relative$ tolerances with $falloff$ parameter determining how fast the absolute tolerance effect decays as the value $y$ grows.

$$
tolerance = \frac{absolute}{falloff \cdot |y| + 1} + relative \cdot |y|
$$
