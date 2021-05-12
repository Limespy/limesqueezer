import numpy as np
import matplotlib.pyplot as plt
import time

def fastcompress(x, y, atol=2e-2, mins = 100):
    '''Fast compression using sampling and splitting from largest error
    x: 1D numpy array
    y: 1D numpy array
    atol: absolute error tolerance
    mins: minimum number of samples, don't change if you don't understand
    '''
    def rec(a, b):
        '''Recurser'''
        n = b-a-1
        step = 1 if n<=mins else round(n / (2*(n - mins)**0.5 + mins))

        err = lambda xf, yf: np.abs((y[b]- y[a]) /(x[b] - x[a])* (xf - x[a]) + y[a] - yf)
        i = a + step*np.argmax(err(x[a+1:b-1:step], y[a+1:b-1:step]))

        return np.concatenate((rec(a, i), rec(i, b)[1:])) if err(x[i], y[i]) > atol else (a,b)

    return rec(0,len(x)-1)

c = 10 # Difficulty of compression of reference

n_data = int(1e5)

# Setting up reference data
x_data = np.linspace(0,1,n_data)
y_data = np.sin(x_data*c*4)+3
# y = 2 - x_data/ (1+c-c*x_data) 

t_start = time.perf_counter()
indices = fastcompress(x_data, y_data)
x_compressed, y_compressed = x_data[indices], y_data[indices]
t = time.perf_counter()-t_start

print("Compression time\t%.3f ms" % (t*1e3))
print("Length of compressed array\t%i"%len(x_compressed))
compression_factor = 1 - len(x_compressed)/len(x_data)
print("Compression factor\t%.3f %%" % (compression_factor*1e2))


plt.figure()
plt.plot(x_data, y_data)
plt.plot(x_compressed, y_compressed,"-o")
plt.show()