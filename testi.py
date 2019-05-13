
































# import numpy as np
# import sympy as sp
# import matplotlib.pyplot as plt
# from scipy import integrate
#
# c_2 = 1 / 14
# A = 1 / 14
# B = 1 / 15
# x_init = 0
# y_init = np.array([B * np.pi, A * np.pi])
# x_k = np.pi
# EPS = 10 ** -4
#
#
# def system(x, y, sym=False):
#     if sym:
#         return [A * sp.Symbol("y1"), -B * sp.Symbol("y0")]
#     return np.array([A * y[1], -B * y[0]])
#
#
# def second_order_runge_kutta(k, local_error=False, x_0=x_init, y_0=y_init):
#     a_2_1 = c_2
#     b_2 = 1 / (2 * c_2)
#     b_1 = 1 - b_2
#     x = []
#     y = []
#     y_nodes = []
#     h = (x_k - x_init) / k
#     if x_k - x_0 < h:
#         h = x_k - x_0
#     if local_error:
#         k = 1
#         counter = 2
#     else:
#         counter = 1
#     for j in range(counter):
#         x = [x_0]
#         y = [y_0]
#         h /= 2 ** j
#         for i in range(k + j):
#             k_1 = h * system(x[i], y[i])
#             k_2 = h * system(x[i] + c_2 * h, y[i] + a_2_1 * k_1)
#             print(k_1, k_2)
#             y.append(y[i] + b_1 * k_1 + b_2 * k_2)
#             x.append(x[i] + h)
#         y_nodes.append(y[-1])
#     if local_error:
#         x_node = x[-1]
#         return x_node, y_nodes
#     return y
#
#
# def runge_error(approx_1, approx_2, method=second_order_runge_kutta):
#     if method == second_order_runge_kutta:
#         accuracy_order = 2
#     else:
#         accuracy_order = 4
#     error = np.linalg.norm((approx_2 - approx_1) / (2 ** accuracy_order - 1))
#     return error
#
#
# def algorithm(method=second_order_runge_kutta):
#     r = 1
#     k = 2
#     approx_1 = method(k)[-1]
#     approx_2 = method(k * (2 ** r))[-1]
#     error = runge_error(approx_1, approx_2, method=method)
#     while error > EPS:
#         r += 1
#         approx_1 = approx_2
#         approx_2 = method(k * (2 ** r))[-1]
#         error = runge_error(approx_1, approx_2, method=method)
#     return approx_2, error
#
#
# def optimal_step(k, method=second_order_runge_kutta):
#     if method == second_order_runge_kutta:
#         accuracy_order = 2
#     else:
#         accuracy_order = 4
#     approx_1 = method(k)[-1]
#     approx_2 = method(k * 2)[-1]
#     h = (x_k - x_init) / k
#     h_opt = (h / 2) * (((2 ** accuracy_order - 1) * EPS)
#                        / np.linalg.norm(approx_2 - approx_1)) ** (1 / accuracy_order)
#     k_opt = int(np.ceil((x_k - x_init) / h_opt))
#     h_opt = (x_k - x_init) / k_opt
#     return h_opt, k_opt
#
#
# print(optimal_step(2))


