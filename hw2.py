import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def p1():
    def bisection(f, a, b, tol):
        x = (a + b) / 2
        while np.abs(b - a) >= tol:
            if np.sign(f(x)) == np.sign(f(a)):
                a = x
            else:
                b = x
            x = (a + b) / 2

        return x

    def f(x, y, beta):
        return np.array([y[1],  -1. * y[0] * (-100 * (np.sin(2. * x) + 1) + beta)])

    def shoot(beta):
        tspan = np.array([x0, xf])
        init_condition = np.array([y0, A])
        sol = solve_ivp(lambda x, y: f(x, y, beta), tspan, init_condition)
        return sol.y[0, -1]

    L = 1
    x0 = -L
    y0 = 0

    xf = L
    yf = 0

    A = 1
    
    beta = 0
    sign = np.sign(shoot(beta))
    dbeta = 0.1

    num_modes = 3
    eigenvals = np.zeros(num_modes)
    eigenfunc = np.zeros(num_modes)
    k = 0

    while k < num_modes:
        beta_next = beta + dbeta
        sign_next = np.sign(shoot(beta_next))
        if sign != sign_next:
            eigenvals[k] = bisection(shoot, beta, beta_next, 1e-8)
            k = k + 1
        beta = beta_next
        sign = sign_next

    for k in range(num_modes):
        # print("eigenval = {}".format(eigenvals[k]))
        # print("Predicted value = {}".format(((k + 1) * np.pi / (2 * L)) ** 2))
        tspan = np.array([x0, xf])
        # x_eval = np.linspace(x0, xf, 1000)
        x_eval = np.array([-1, 0, 1])
        init_condition = np.array([y0, A])
        sol = solve_ivp(lambda x, y: f(x, y, eigenvals[k]),
                        tspan, init_condition,
                        t_eval=x_eval)
        x = sol.t
        y = sol.y[0, :]
        eigenfunc[k] = y[1]
        # plt.plot(x, y)

    # plt.show()

    t_A1 = eigenvals[0]
    t_A2 = eigenfunc[0]

    t_A3 = eigenvals[1]
    t_A4 = eigenfunc[1]

    t_A5 = eigenvals[2]
    t_A6 = eigenfunc[2]
    return t_A1, t_A2, t_A3, t_A4, t_A5, t_A6
#------------------------------------------------------------------------------- 

def p2_trap():
    # x''(t) = x = f(x, t)
    # x(0) = 1
    # x'(0) = 0
    # t0 = 0, tN = 1, dt = 0.1

    def f(x, t):
        return x

    def true_solution(t):
        return 0.5 * (np.exp(t) + np.exp(-t)) 

    p0 = 1
    v0 = 0
    t0 = 0
    tN = 1
    dtlist = np.array([0.1, 0.01])
    sollist = np.zeros_like(dtlist)
    errlist = np.zeros_like(dtlist)
    for i in range(len(dtlist)):
        dt = dtlist[i]
        t = np.arange(t0, tN + dt / 2, dt)
        p = np.zeros(len(t))
        v = np.zeros(len(t))
        p[0] = p0
        v[0] = v0

        for k in range(len(t) - 1):
            ak = f(p[k], t[k])
            v[k + 1] = (dt * ak + v[k] * ( 1. + 0.25 * dt**2)) / (1. - 0.25 * dt **2)
            p[k + 1] = p[k] + 0.5 * dt * (v[k] + v[k + 1])

        # tplot = np.linspace(t0, tN, 1000)
        # xplot = true_solution(tplot)
        # plt.plot(tplot, xplot, 'k')
        # plt.plot(t, p, 'ro')

        # plt.show()
        # print(p[-1])
        sollist[i] = p[-1]

        # err = np.max(np.abs(p - true_solution(t)))
        # print(err)

        global_err = np.abs(p[-1] - true_solution(t)[-1])
        errlist[i] = global_err

    t_A7 = sollist[0]
    t_A8 = errlist[0]

    t_A9 = sollist[1]
    t_A10 = errlist[1]

    return t_A7, t_A8, t_A9, t_A10
#------------------------------------------------------------------------------- 

def p2_midp():
    # x''(t) + x = 0
    # x(0) = 1
    # x'(0) = 0
    # t0 = 0, tN = 1, dt = 0.1

    def f(x, t):
        return -1. * x

    def true_solution(t):
        return np.cos(t) 
    
    p0 = 1
    v0 = 0
    t0 = 0
    tN = 1
    dtlist = np.array([0.1, 0.01])
    sollist = np.zeros_like(dtlist)
    errlist = np.zeros_like(dtlist)
    for i in range(len(dtlist)):
        dt = dtlist[i]
        t = np.arange(t0, tN + dt / 2, dt)
        p = np.zeros(len(t))
        v = np.zeros(len(t))
        p[0] = p0
        v[0] = v0

        p[1] = p0 + dt * v0
        a0 = f(p0, t0)
        v[1] = v0 + dt * a0 
        for k in range(1, len(t) - 1):
            p[k + 1] = p[k - 1] + 2. * dt * v[k]
            ak = f(p[k], t[k])
            v[k + 1] = v[k - 1] + 2. * dt * ak

        # tplot = np.linspace(t0, tN, 1000)
        # xplot = true_solution(tplot)
        # plt.plot(tplot, xplot, 'k')
        # plt.plot(t, p, 'ro')

        # plt.show()
        # print(p[-1])
        sollist[i] = p[-1]

        # err = np.max(np.abs(p - true_solution(t)))
        # print(err)

        global_err = np.abs(p[-1] - true_solution(t)[-1])
        errlist[i] = global_err

    t_A11 = sollist[0]
    t_A12 = errlist[0]

    t_A13 = sollist[1]
    t_A14 = errlist[1]
    
    return t_A11, t_A12, t_A13, t_A14
#------------------------------------------------------------------------------- 

def p3():
    # y'' + p(x)*y' + q(x)*y = r(x)
    # (1 - x^2)y'' - x * y' + al^2 * y = 0
    # y'' - ((x)/(1 - x^2)) * y' + ((al^2)/(1 - x^2)) * y = 0
    # alpha = 1.; y(-0.5) = -0.5 and y(0.5) = 0.5
    # alpha = 2.; y(-0.5) = 0.5 and y(0.5) = 0.5
    # alpha = 3.; y(-0.5) = -1/3 and y(0.5) = 1/3

    def true_sol(x, a):
        if (a == 0):
            return x
        elif (a == 1):
            return 1. - (2. * (x**2))
        elif (a == 2):
            return x - ((4./3.)*(x**3))

    alphas = np.array([1., 2., 3.])
    x0 = -0.5
    y0list = np.array([-0.5, 0.5, -1./3.])
    xN = 0.5
    yNlist = np.array([0.5, 0.5, 1./3.]) 

    dx = 0.1
    x = np.arange(x0, xN + 0.5 * dx, dx)
    N = len(x)

    sollist = np.zeros_like(alphas)
    errlist = np.zeros_like(alphas)

    for a in range(len(alphas)):
        al = alphas[a]
        y0 = y0list[a]
        yN = yNlist[a]

        p = (-1. * x) / (1. - x**2) 
        q = (al**2) / (1. - x**2) 
        r = np.zeros_like(x)

        A = np.zeros((N, N))
        b = np.zeros((N, 1))

        A[0, 0] = 1
        b[0] = y0
        A[N-1, N-1] = 1
        b[N-1] = yN

        for k in range(1, N - 1):
            A[k, k-1] = (1 - dx * p[k] / 2)
            A[k, k] = (-2 + dx ** 2 * q[k])
            A[k, k + 1] = (1 + dx * p[k] / 2)

            b[k] = - (dx**2) * r[k]

        y = np.linalg.solve(A, b).reshape(N)

        # plt.plot(x, true_sol(x), 'k')
        # plt.plot(x, y, 'b', x0, y0, 'ro', xN, yN, 'ro')
        # plt.show()

        err = np.max(np.abs(y - true_sol(x, a)))
        # print("Error = {}".format(err))
        # print(y[N//2])

        # global_err = np.abs(y[-1] - true_sol(x, a)[-1])
        errlist[a] = err
        sollist[a] = y[N//2]

    t_A15 = sollist[0]
    t_A16 = errlist[0]

    t_A17 = sollist[1]
    t_A18 = errlist[1]

    t_A19 = sollist[2]
    t_A20 = errlist[2]

    return t_A15, t_A16, t_A17, t_A18, t_A19, t_A20
#------------------------------------------------------------------------------- 

A1, A2, A3, A4, A5, A6 = p1()
# print("A1", A1)
# print("A2", A2)
# print("A3", A3)
# print("A4", A4)
# print("A5", A5)
# print("A6", A6)

A7, A8, A9, A10 = p2_trap()
# print("A7", A7)
# print("A8", A8)
# print("A9", A9)
# print("A10", A10)

A11, A12, A13, A14,= p2_midp()
# print("A11", A11)
# print("A12", A12)
# print("A13", A13)
# print("A14", A14)

A15, A16, A17, A18, A19, A20 = p3()
# print("A15", A15)
# print("A16", A16)
# print("A17", A17)
# print("A18", A18)
# print("A19", A19)
# print("A20", A20)