# interp-compression
Lossy compression of numerical data using interpolation


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

$$

$$

$$

$$

$$