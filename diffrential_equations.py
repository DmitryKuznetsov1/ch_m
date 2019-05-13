import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import integrate

KSI = 1/14
A = 1/14
B = 1/15
X_0 = 0
Y_0 = np.array([B * np.pi, A * np.pi])
X_k = np.pi
EPS = 10 ** -4


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
        x.append(x[i] + h)
        y.append(y[i] + b_1 * k_1 + b_2 * k_2)
    return y


def opponent(k):
    return two_stage_schema(k, c_2=1)


def runge_error(y_dash, y_wave, s=2):
    error = np.linalg.norm((y_wave - y_dash) / (1 - 2 ** -s))
    return error


def solving(method=two_stage_schema, epsilon=EPS):
    k = 2
    y_dash = method(k)[-1]
    k *= 2
    y_wave = method(k)[-1]
    error = runge_error(y_dash, y_wave)
    while error > epsilon:
        k *= 2
        y_dash = y_wave
        y_wave = method(k)[-1]
        error = runge_error(y_dash, y_wave)
    return k, y_dash, error


def step_opt(s=2, epsilon=EPS, method=two_stage_schema):
    k = 2
    y_dash = method(k)[-1]
    y_wave = method(2 * k)[-1]
    h = (X_k - X_0) / k
    h_opt = (h / 2) * ((2 ** s - 1) * epsilon / np.linalg.norm(y_dash - y_wave)) ** (1 / s)
    print(h_opt)
    k_opt = int(np.ceil((X_k - X_0) / h_opt)) - 1
    h_opt = (X_k - X_0) / k_opt
    return k_opt, h_opt


def integrate_auto_step(method=two_stage_schema, delta=10**-5):
    x_list = [X_0]
    y_list = [Y_0]
    k = 2
    len_segmenth = X_k - X_0
    steps_list = [k]
    while X_k > x_list[-1]:
        pass
    return 0


def main():
    true_result = integrate.solve_ivp(
        f, (X_0, X_k), Y_0, dense_output=True, atol=1e-13, rtol=1e-13).sol.__call__(X_k)
    my_result, error = solving()[1:3]
    k_opt, h_opt = step_opt()
    y_nodes = two_stage_schema(k_opt)
    y_nodes_er = two_stage_schema(k_opt*2)[::2]
    my_result_opt = y_nodes[-1]
    x_nodes = [X_0 + h_opt * i for i in range(k_opt + 1)]
    errors = [runge_error(y_nodes[i], y_nodes_er[i]) for i in range(len(x_nodes))]
    plt.plot(x_nodes, errors)
    plt.title('runge error by x')
    plt.grid(True)
    plt.show()
    print('True result:', true_result, '\nMy result:', my_result, '\nAbsolute error:', errors[-1],
          '\nEstimated error:', error, '\nOptimal step:', h_opt, '\nMy result with optimal step:', my_result_opt)
    return 0


main()
