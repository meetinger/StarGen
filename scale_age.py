import math
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def f(x):
    return math.log10(x)


a = 1
b = 11.5e+9
eps = 3e+9
k = 2.1
x_ages = list(np.arange(0, b, eps))
x_ages[0] = 1
print(len(x_ages))
# x_ages = [a, 1e+9, 2e+9, 3e+9, 4e+9, 11e+9, 12e+9, 13e+9, 27e+9, 28e+9, 29e+9, b]
# y_ages = [0, 100,   200,  275,  300,   450,   500,   525,    725,   800,   900, 1000]
tmp = [f(x_ages[i]) for i in range(0, len(x_ages) // 2)]
y_ages = tmp + [-i + k * tmp[len(x_ages) // 2 - 1] for i in reversed(tmp)]
print(len(y_ages))


def test():
    x_for_plot = list(np.arange(a, b, eps / 10))

    plt.scatter(x_ages, y_ages, label='Points', color='blue')

    spline = interpolate.splrep(x_ages, y_ages, s=0)

    y_interpolated = interpolate.splev(x_for_plot, spline, der=0)

    spline_inv = interpolate.splrep(y_ages, x_ages, s=0)

    x_calc = interpolate.splev(y_interpolated, spline_inv, der=0)

    print(np.array(x_for_plot))
    print(x_calc)

    plt.plot(x_for_plot, y_interpolated, label='Interpolated')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


test()
