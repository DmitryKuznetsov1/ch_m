import math
from math import cos, sin, exp, ceil
from sympy import symbols
from scipy import integrate, optimize
import numpy as np
import sympy as sp

A = 2.1
B = 3.3
ALPHA = 0.4
BETA = 0
EPS = 10**-6
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
    # print('nodes', nc_nodes)
    nc_moments = moments(n, a, b)
    # print('moments', nc_moments)
    matrix_of_nodes = nodes_degrees(nc_nodes)
    # print('matrix of modes', matrix_of_nodes)
    quadrature_formula_coeffs = np.linalg.solve(matrix_of_nodes, nc_moments)
    # print(quadrature_formula_coeffs)
    my_solve_ = my_solve(quadrature_formula_coeffs, nc_nodes)
    if value:
        return my_solve_
    # print('my_solve =', my_solve_)
    # print('true_solve =', integrate.quad(source_f, a, b)[0])
    # print('error =', abs(my_solve(quadrature_formula_coeffs, nc_nodes) - integrate.quad(source_f, a, b)[0]))
    # print('methodical_error <=', methodical_error(a, b, nc_nodes, 1))
    return nc_nodes, quadrature_formula_coeffs


# nc = newton_cotes(A, B, N)
# print(nc)


def gauss(a, b, n, value=True):
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
    my_solve_ = my_solve(gauss_coeffs, gauss_nodes)
    if value:
        return my_solve_
    # print('gauss coeffs', gauss_coeffs)
    # print('gauss my solve', my_solve(gauss_coeffs, gauss_nodes))
    # print('gauss error =', abs(my_solve(gauss_coeffs, gauss_nodes) - integrate.quad(source_f, a, b)[0]))
    # print('gauss methodical error <=', methodical_error(a, b, gauss_nodes, 2))
    return np.flip(gauss_nodes, 0), np.flip(gauss_coeffs, 0)


# gauss_ = gauss(A, B, N)
# print(gauss_)


def s_h_sum(a, b, n, k, method=newton_cotes):
    s_h = 0
    h = (b - a) / k
    for i in range(k):
        s_h += method(a + i * h, a + (i + 1) * h, n, value=True)
    return s_h


# s_h_sum = s_h_sum(A, B, N, 32)
# true = integrate.quad(source_f, A, B, epsabs=1e-10)[0]


def runge(a, b, n, k=2, epsilon=10**-6, method=newton_cotes, accuracy=True):
    l = 2
    m = n - 1
    h = (b - a) / k
    s_h1, s_h2 = s_h_sum(a, b, n, k, method=method), s_h_sum(a, b, n, k * l, method=method)
    if accuracy:
        error = abs((s_h1 - s_h2) / (1 - l ** (-m)))
        # true_error = abs(integrate.quad(source_f, a, b, epsabs=1e-10)[0] - s_h1)
        return error
    h_opt = 0.95 * h * ((epsilon * (1 - l ** (-m))/abs(s_h1 - s_h2)) ** (1 / m))
    k_opt = math.ceil((b - a) / h_opt)
    h_opt = (b - a) / k_opt
    return h_opt, k_opt


print('Runge error for k=2:', runge(A, B, N))
h_opt_, k_opt_ = runge(A, B, N, accuracy=False)
print('Runge h opt, k opt:', h_opt_, k_opt_)
print('Runge error for k opt:', runge(A, B, N, k=k_opt_))


def richardson(a, b, n, r=2, epsilon=10**-6, method=newton_cotes, accuracy=True):
    l = 2
    m = n - 1
    h = (b - a) / r
    error = epsilon + 1
    if accuracy:
        h_matrix = np.ones((r + 1, r + 1)) * (-1)
        s_h_vector = []
        for i in range(r + 1):
            for j in range(r):
                h_matrix[i][j] = (h / l**i) ** (m + j)
            s_h_vector.append(-s_h_sum(a, b, n, r * l ** i, method=method))
        c_m = np.linalg.solve(h_matrix, s_h_vector)
        # print(h_matrix, s_h_vector)
        # print(c_m)
        error = abs(c_m[-1] + s_h_vector[0])
        return error, h
    r = 1
    s_h_vector = [- s_h_sum(a, b, n, r * l ** 0, method=method), - s_h_sum(a, b, n, r * l ** r, method=method)]
    while error > epsilon:
        r += 1
        h_matrix = np.ones((r + 1, r + 1))*(-1)
        for i in range(r + 1):
            for j in range(r):
                h_matrix[i][j] = (h / l ** i) ** (m + j)
        s_h_vector.append(-s_h_sum(a, b, n, r * l ** r, method=method))
        c_m = np.linalg.solve(h_matrix, s_h_vector)
        h_matrix[-1][-1] = 0
        error = abs(np.dot(h_matrix[-1], c_m))
    print('r:', r)
    return c_m[-1], error, (b-a)*r/(l**r)


R_h1, h1 = richardson(A, B, N, accuracy=False)[1], richardson(A, B, N, accuracy=False)[2]
print('Error, step:', R_h1, h1)
h_opt_1 = h1 * (EPS/abs(R_h1))**(1 / (N - 1))
print('h opt1:', h_opt_1)
k_opt_1 = math.ceil((B-A)/h_opt_1)
print('k opt1:', k_opt_1)
print('Error with h opt1:', richardson(A, B, N, r=k_opt_1, accuracy=True)[0])


# for i in range(6):
#     print('Estimated error for r = 6 is', richardson(A, B, N, r=i+1, epsilon=10**-6, method=newton_cotes, accuracy=True))


def aitken(a, b, n, method=newton_cotes):
    k = 3
    l = 2
    s_h1, s_h2, s_h3 = s_h_sum(a, b, n, k, method=method), s_h_sum(a, b, n, k * l, method=method),\
                       s_h_sum(a, b, n, k * l ** 2, method=method)
    m = -(np.log(abs((s_h3 - s_h2) / (s_h2 - s_h1))) / np.log(l))
    return m


print('m from aitken', aitken(A, B, N, method=newton_cotes))


def comp_quadr_formula(a, b, n, epsilon=10**-6, method=newton_cotes, accuracy_rule=runge):
    true_solve = integrate.quad(source_f, A, B, epsabs=1e-10)[0]
    if accuracy_rule == runge:
        h_opt, k_opt = accuracy_rule(a, b, n, epsilon=10**-6, method=method, accuracy=False)
        est_error = accuracy_rule(a, b, n, k_opt, epsilon=10**-6, method=method, accuracy=True)
        my_solve_ = s_h_sum(a, b, n, k_opt, method=method)
    elif accuracy_rule == richardson:
        my_solve_, est_error = accuracy_rule(a, b, n, epsilon=epsilon, method=method, accuracy=False)
    print('Composite solve', my_solve_)
    print('True solve', true_solve)
    print('Estimated error', est_error)
    print('True error', abs(true_solve - my_solve_))


comp_quadr_formula(A, B, N, epsilon=EPS, method=newton_cotes, accuracy_rule=runge)


