import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import integrate

c_2 = 1 / 14
A = 1 / 14
B = 1 / 15
x_init = 0
y_init = np.array([B * np.pi, A * np.pi])
x_k = np.pi


def system(x, y, sym=False):
    if sym:
        return [A * sp.Symbol("y1"), -B * sp.Symbol("y0")]
    return np.array([A * y[1], -B * y[0]])


def second_order_runge_kutta(k, local_error=False, x_0=x_init, y_0=y_init):
    a_2_1 = c_2
    b_2 = 1 / (2 * c_2)
    b_1 = 1 - b_2
    x = []
    y = []
    y_nodes = []
    h = (x_k - x_init) / k
    if x_k - x_0 < h:
        h = x_k - x_0
    if local_error:
        k = 1
        counter = 2
    else:
        counter = 1
    for j in range(counter):
        x = [x_0]
        y = [y_0]
        h /= 2 ** j
        for i in range(k + j):
            k_1 = h * system(x[i], y[i])
            k_2 = h * system(x[i] + c_2 * h, y[i] + a_2_1 * k_1)
            print(k_1, k_2)
            y.append(y[i] + b_1 * k_1 + b_2 * k_2)
            x.append(x[i] + h)
        y_nodes.append(y[-1])
    if local_error:
        x_node = x[-1]
        return x_node, y_nodes
    return y


print(second_order_runge_kutta(2))