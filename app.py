import math
from math import cos, sin, exp
from sympy import symbols
from scipy import integrate, optimize
import numpy as np
import sympy as sp


A = 2.1
B = 3.3
ALPHA = 0.4
BETA = 0
N = 3   # number of nodes
x = symbols('x')
source_f = lambda x: (4.5 * cos(7 * x) * exp(-2 * x / 3) + 1.4 * sin(1.5 * x) * exp(-x / 3) + 3) * weight_func(x, 0)


def func(x):
    return 4.5 * cos(7 * x) * exp(-2 * x / 3) + 1.4 * sin(1.5 * x) * exp(-x / 3) + 3


def func_sym():
    x_ = sp.Symbol("x")
    return 4.5 * sp.cos(7 * x_) * sp.exp(-2 * x_ / 3) + 1.4 * sp.sin(1.5 * x_) * sp.exp(-x_ / 3) + 3


def weight_func(x, j):
    return ((x - A) ** -ALPHA) * ((B - x) ** -BETA) * x ** j


def node_pol(x, nodes, j=-1):
    pol = 1
    for i in range(len(nodes)):
        if i != j:
            pol *= (x - nodes[i])
    return pol


def abs_p_omega(x, nodes, omega_degree, a, b):
    return abs(weight_func(x, 0) * node_pol(x, nodes) ** omega_degree)


def moments(number_of_nodes, a, b):
    mu = np.zeros(number_of_nodes)
    for i in range(number_of_nodes):
        f = lambda x: x ** i * weight_func(x, 0)    # under integral function
        mu[i] = integrate.quad(f, a, b)[0]
    return mu


def nodes_degrees(array_of_nodes):
    k = len(array_of_nodes)
    x = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            x[i][j] = array_of_nodes[j] ** i
    return x


def my_solve(coeffs, nodes):
    integral = 0
    for i in range(N):
        integral += coeffs[i] * func(nodes[i])
    return integral


def methodical_error(a, b, nodes, degree):
    f = -sp.Abs(sp.diff(func_sym(), sp.Symbol('x'), degree * len(nodes)))
    f = sp.utilities.lambdify(sp.Symbol('x'), f)
    return (-optimize.minimize_scalar(f, bounds=(a, b), method='Bounded').fun / math.factorial(degree * len(nodes))) * \
           integrate.quad(abs_p_omega, a, b, args=(nodes, degree, a, b,))[0]


def newton_cotes(a, b, n, value=True):
    nc_nodes = np.linspace(a, b, n)
    print(nc_nodes)
    # print('nodes', nc_nodes)
    nc_moments = moments(n, a, b)
    # print('moments', nc_moments)
    matrix_of_nodes = nodes_degrees(nc_nodes)
    # print('matrix of modes', matrix_of_nodes)
    quadrature_formula_coeffs = np.linalg.solve(matrix_of_nodes, nc_moments)
    # print(quadrature_formula_coeffs)
    my_solve_ = my_solve(quadrature_formula_coeffs, nc_nodes)
    if value:
        print(my_solve_)
        return my_solve_
    # print('my_solve =', my_solve_)
    # print('true_solve =', integrate.quad(source_f, a, b)[0])
    # print('error =', abs(my_solve(quadrature_formula_coeffs, nc_nodes) - integrate.quad(source_f, a, b)[0]))
    # print('methodical_error <=', methodical_error(a, b, nc_nodes, 1))
    return nc_nodes, quadrature_formula_coeffs


# nc = newton_cotes(A, B, N)
# print(nc)


def gauss(a, b, n):
    gauss_moments = moments(2 * n, a, b)
    # print('moments\n', gauss_moments)
    mu_sj = np.zeros((n, n))
    for s in range(n):
        for j in range(n):
            mu_sj[s, j] = gauss_moments[s + j]
    # print('moments matrix\n', mu_sj)
    mu_ns = gauss_moments[n:]
    # print('moments vector n to 2n -1 ', mu_ns)
    a_coeffs = list(np.ravel(np.linalg.solve(mu_sj, -mu_ns)))
    a_coeffs.append(1)
    a_coeffs.reverse()
    # print('polinomial coeffs', a_coeffs)
    gauss_nodes = np.roots(a_coeffs)
    # print('gauss nodes', gauss_nodes)
    gauss_matrix_of_nodes = nodes_degrees(gauss_nodes)
    # print('matrix of nodes\n', gauss_matrix_of_nodes)
    mu_s = gauss_moments[:n]
    # print('moments vector 0 to n - 1', mu_s)
    gauss_coeffs = np.linalg.solve(gauss_matrix_of_nodes, mu_s)
    # print('gauss coeffs', gauss_coeffs)
    # print('gauss my solve', my_solve(gauss_coeffs, gauss_nodes))
    # print('gauss error =', abs(my_solve(gauss_coeffs, gauss_nodes) - integrate.quad(source_f, a, b)[0]))
    # print('gauss methodical error <=', methodical_error(a, b, gauss_nodes, 2))
    return np.flip(gauss_nodes, 0), np.flip(gauss_coeffs, 0)


# gauss_ = gauss(A, B, N)
# print(gauss_)


def s_h_values(a, b, n, k, method=newton_cotes):
    s_h = 0
    h = (b - a) / k
    print(k, h)
    for i in range(k):
        print(i)
        s_h += method(a + i * h, a + (i + 1) * h, n, value=True)
    return s_h


my = s_h_values(A, B, N, 423, method=newton_cotes)
print('my', my)
print('true_solve =', integrate.quad(source_f, A, B)[0])


def runge(a, b, n, k=2, epsilon=10**-6, method=newton_cotes, accuracy=True):
    l_ = 2
    m = n - 1
    h = (b - a) / k
    s_h1, s_h2 = s_h_values(a, b, n, k, method=method), s_h_values(a, b, n, k * l_, method=method)
    if accuracy:
        error = abs((s_h1 - s_h2) / (1 - l_ ** (-m)))
        return error
    h_opt = 0.95 * h * ((epsilon * (1 - l_ ** (-m))/abs(s_h1 - s_h2)) ** (1 / m))
    print(s_h1 - s_h2)
    k_opt = math.ceil((b - a) / h_opt)
    h_opt = (b - a) / k_opt
    return h_opt, k_opt


# print(runge(A, B, N, accuracy=False))