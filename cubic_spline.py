import math

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def f(x):
    return math.log10(x)


def inv_spline(spline):
    print(spline)
    roots = interpolate.sproot(spline)
    print(roots)

def compare_func(func, a, b, eps):
    x_for_plot = list(np.arange(a, b, eps / 10))
    x = list(np.arange(a, b, eps))

    y_for_plot = [func(i) for i in x_for_plot]
    y = [func(i) for i in x]

    plt.plot(x_for_plot, y_for_plot, label='Orig')
    plt.scatter(x, y, label='Points', color='blue')

    spline = interpolate.splrep(x, y, s=0)

    y_interpolated = interpolate.splev(x_for_plot, spline, der=0)

    inv_spline(spline)

    spline_inv = interpolate.splrep(y, x, s=0)
    # spline_inv = inv_spline(spline)
    #
    x_calc = interpolate.splev(y_interpolated, spline_inv, der=0)


    print(x_for_plot)
    print(x_calc)
    # x_calc and x must be equal

    plt.plot(x_for_plot, y_interpolated, label='Interpolated')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


compare_func(f, 1, 100, 1)
