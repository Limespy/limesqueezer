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
