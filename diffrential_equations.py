import numpy as np
import sympy as sp

KSI = 1/14
A = 1/14
B = 1/15
X_0 = 0
Y_0 = np.array([B * np.pi, A * np.pi])
X_k = np.pi


def f(x, y, symbol=False):
    if symbol:
        return [A * sp.Symbol('y1'), -B * sp.Symbol('y0')]
    return np.array([A * y[1], -B * y[0]])


def two_stage_schema(k, x_0=X_0, y_0=Y_0, x_k=X_k, c_2=KSI):
    x, y = [x_0], [y_0]
    b_2 = 1 / (2 * c_2)
    b_1 = 1 - b_2
    h = (x_k - x_0) / k
    for i in range(k):
        k_1 = h * f(x[i], y[i])
        k_2 = h * f(x[i] + c_2 * h, y[i] + c_2 * k_1)
        print(k_1, k_2)
        x.append(x[i] + h)
        y.append(y[i] + b_1 * k_1 + b_2 * k_2)
    return y




