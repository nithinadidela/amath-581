"""
Nithin Adidela
AMATH 581
Homework 0
yyyy-mm-dd
2023-10-16
"""

import numpy as np
import matplotlib.pyplot as plt

# Problem 1
p1_A = np.array([[2.1, 3.8], [-1.6, 0.]])
# print(p1_A)
A1 = p1_A

p1_b = np.array([[2.], 
              [3.]])
# print(p1_b)
A2 = p1_b

p1_c = np.dot(p1_A, p1_b)
# print(p1_c)
A3 = p1_c

p1_x = np.linalg.solve(p1_A, p1_b)
# print(p1_x)
A4 = p1_x

# Problem 2
p2_x = np.array([1, 3, 4, 8, 9])
p2_y = np.array([3, 4, 5, 7, 12])
p2_m, p2_c = np.polyfit(p2_x, p2_y, 1)
p2_fit_line = p2_m * p2_x + p2_c

plt.plot(p2_x, p2_y)
plt.plot(p2_x, p2_fit_line)
plt.show()
A5 = p2_m
A6 = p2_c

# Problem 3
def p3_f(t_x):
    return np.tan(t_x) - t_x

def p3_fx(t_x):
    return (1. / (np.cos(t_x)**2)) - 1.

# Bisection method to find the root
def p3_bisection_method(t_f, t_a, t_b, t_tol):
    count = 1
    a = t_a
    b = t_b
    x = (t_a + t_b) / 2.
    while(t_tol < np.abs(t_f(x))):
        count += 1
        if((t_f(a) * t_f(x)) < 0):
            b = x
        else:
            a = x
        x = (a + b) / 2.

    return x, count

# Newton method to find the root
def p3_Newton_method(t_f, t_fx, t_x0, t_tol):
    count = 1
    x = t_x0
    while(t_tol < np.abs(t_f(x))):
        count += 1
        x -= t_f(x) / t_fx(x)

    return x, count


p3_a = 2.
p3_b = 4.6
p3_root_bm, p3_count_bm = p3_bisection_method(p3_f, p3_a, p3_b, 1e-8)
A7 = p3_root_bm
A8 = p3_count_bm

p3_x0 = 4.3
p3_root_nm, p3_count_nm = p3_Newton_method(p3_f, p3_fx, p3_x0, 1e-8)
A9 = p3_root_nm
A10 = p3_count_nm

