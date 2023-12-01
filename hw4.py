import numpy as np
import scipy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def p1(Nt, method):
    # u_t = 5 * u_xx
    # u(t, -1) = 0
    # u(t, 2) = 0
    # u(0, x) = sin(4./3. *  pi * (x + 1))
    # u(t, x) = e^(-5.16/9. * pi^2  * t) * sin(4./3. * pi * (x + 1))

    Nx = 25
    x0 = -1.
    xf = 2.
    x = np.linspace(x0, xf, Nx)
    dx = x[1] - x[0]
    # print("Nx = {}".format(Nx))
    # print("dx = {}".format(dx))

    t0 = 0
    tf = 0.25
    # Nt = 2 * (Nx - 1) ** 2 + 1
    t = np.linspace(t0, tf, Nt)
    dt = t[1] - t[0]
    # print("Nt = {}".format(Nt))
    # print("dt = {}".format(dt))
    # print("dt/dx^2 = {}".format(dt / dx ** 2))

    U = np.zeros((Nx, Nt))
    U[:, 0] = np.sin((4./3.) * np.pi * (x + 1))
    U[0, :] = 0
    U[-1, :] = 0

    A = 5. * (np.diag(-2 * np.ones(Nx - 2)) + np.diag(np.ones(Nx - 3), 1) + np.diag(np.ones(Nx - 3), -1)) / dx ** 2

    # Check the eigenvalues of A
    # vals, _ = np.linalg.eig(A)
    # k = np.arange(1, Nx - 1)
    # exact_vals = (2 / dx ** 2) * (np.cos(k * np.pi * dx) - 1)
    # print("Eigenvalues of A = {}".format(sorted(vals)))
    # print("Exact eigenvalues of A = {}".format(sorted(exact_vals)))
    # print("dt * lambda = {}".format(sorted(dt * vals)))

    def f(t, u):
        return A @ u

    if (method == "FE"):
        # Solve with Forward Euler
        for k in range(Nt - 1):
            U[1:-1, (k + 1):(k + 2)] = U[1:-1, k:(k + 1)] + dt * f(t[k], U[1:-1, k:(k + 1)])
    elif (method == "BE"):
        # Backward Euler
        for k in range(Nt - 1):
            U[1:-1, (k + 1):(k + 2)] = np.linalg.solve(np.eye(Nx - 2) - dt * A, U[1:-1, k:(k + 1)])

    T, X = np.meshgrid(t, x)

    # ax = plt.axes(projection='3d')
    # ax.plot_surface(T, X, U)
    # plt.show()

    def true_solution(t, x):
        return np.exp(-((5. * 16/9.) * np.pi ** 2) * t) * np.sin((4./3.) * np.pi * (x + 1.))

    err = U - true_solution(T, X)
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(T, X, err)
    # plt.show()

    global_err = np.max(np.abs(err))
    # print("Global Error = {}".format(global_err))

    return U[Nx//3, -1], global_err
#------------------------------------------------------------------------------- 

def p2(Nt, method):
    # u_t = u_xx + 10 * x
    # u(t, 0) = 0
    # u(t, 1) = 10t
    # u(0, x) = sin(pi * x) - 0.8 * sin(3. * pi * x)
    # u(t, x) = 10 * t * x + e^(-1. * pi^2  * t) * sin(pi * x) - 0.8 * e^(-9. * pi^2  * t) * sin(3 * pi * x)

    Nx = 21
    x0 = 0.
    xf = 1.
    x = np.linspace(x0, xf, Nx)
    dx = x[1] - x[0]
    # print("Nx = {}".format(Nx))
    # print("dx = {}".format(dx))

    t0 = 0
    tf = 0.1
    # Nt = 2 * (Nx - 1) ** 2 + 1
    t = np.linspace(t0, tf, Nt)
    dt = t[1] - t[0]
    # print("Nt = {}".format(Nt))
    # print("dt = {}".format(dt))
    # print("dt/dx^2 = {}".format(dt / dx ** 2))

    U = np.zeros((Nx, Nt))
    U[:, 0] = np.sin(np.pi * x) - 0.8 * np.sin(3. * np.pi * x)
    U[0, :] = 0
    # U[-1, :] = 0

    A = (np.diag(-2 * np.ones(Nx - 2)) + np.diag(np.ones(Nx - 3), 1) + np.diag(np.ones(Nx - 3), -1)) / dx ** 2

    # Check the eigenvalues of A
    # vals, _ = np.linalg.eig(A)
    # k = np.arange(1, Nx - 1)
    # exact_vals = (2 / dx ** 2) * (np.cos(k * np.pi * dx) - 1)
    # print("Eigenvalues of A = {}".format(sorted(vals)))
    # print("Exact eigenvalues of A = {}".format(sorted(exact_vals)))
    # print("dt * lambda = {}".format(sorted(dt * vals)))

    def f(t, u):
        return A @ u

    c = np.zeros((Nx-2, 1))
    c[:, 0] = 10. * x[1:-1]
    
    if (method == "FE"):
        # Solve with Forward Euler
        for k in range(Nt - 1):
            c[-1, 0] = 10. * x[-2] + ((10. * k * dt)/(dx**2))
            U[1:-1, (k + 1):(k + 2)] = U[1:-1, k:(k + 1)] + dt * ( f(t[k], U[1:-1, k:(k + 1)]) + c)
            U[-1, (k+1):(k+2)] = 10. * (k+1) * dt

    elif (method == "TRAP"):
        I = np.eye(Nx - 2)
        A_lhs = (I - (dt / 2) * A)
        A_rhs = (I + (dt / 2) * A)
        for k in range(Nt - 1):
            c[-1, 0] = 10. * x[-2] + ((10. * 0.5 * (k + k+1) * dt)/(dx**2))
            U[1:-1, (k + 1):(k + 2)] = np.linalg.solve(A_lhs, A_rhs @ U[1:-1, k:(k + 1)] + dt * c)
            U[-1, (k+1):(k+2)] = 10. * (k + 1) * dt

    T, X = np.meshgrid(t, x)

    # ax = plt.axes(projection='3d')
    # ax.plot_surface(T, X, U)
    # plt.show()

    def true_solution(t, x):
        return (10. * t * x) + np.exp((-1. * np.pi ** 2) * t) * np.sin(np.pi * x) \
                            - 0.8 * np.exp((-9. * np.pi ** 2) * t) * np.sin(3. * np.pi * x) 

    err = U - true_solution(T, X)
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(T, X, err)
    # plt.show()

    global_err = np.max(np.abs(err))
    # print("Global Error = {}".format(global_err))

    return U[Nx//2, -1], global_err
#------------------------------------------------------------------------------- 

def p3(Nt, method):
    # u_t = k * (u_xx + u_yy)
    # u(t, 0, y) = 0
    # u(t, 1, y) = 0
    # u(t, x, 0) = 0 
    # u(t, x, 1) = 0 
    # u(0, x, y) = 0; x=y=0 
    # u(t, x) = exp(-10((x-0.5)^2 + (y-0.5)^2)) 

    Nx = 11
    x0 = 0.
    xf = 1.
    x = np.linspace(x0, xf, Nx)
    y = np.linspace(x0, xf, Nx)
    dx = x[1] - x[0]
    # print("Nx = {}".format(Nx))
    # print("dx = {}".format(dx))

    t0 = 0
    tf = 1.0
    # Nt = 2 * (Nx - 1) ** 2 + 1
    t = np.linspace(t0, tf, Nt)
    dt = t[1] - t[0]
    # print("Nt = {}".format(Nt))
    # print("dt = {}".format(dt))
    # print("dt/dx^2 = {}".format(dt / dx ** 2))

    N_total = Nx * Nx
    u = np.zeros((N_total, Nt))
    
    A = np.zeros((N_total, N_total))
    diff = 0.1

    def f(t, u):
        return A[1:-1, 1:-1] @ u

    def point2ind(m, n):
        return n * Nx + m
    
    for n in range(Nx):
        for m in range(Nx):
            k = point2ind(m, n)
            if n == 0:
                A[k, k] = 1
            elif n == Nx-1:
                A[k, k] = 1
            elif m == 0:
                A[k, k] = 1
            elif m == Nx-1:
                A[k, k] = 1
            else:
                A[k, k] =      diff * (-4. / dx ** 2)
                A[k, k + 1] =  diff * ( 1. / dx ** 2) 
                A[k, k - 1] =  diff * ( 1. / dx ** 2) 
                A[k, k + Nx] = diff * ( 1. / dx ** 2) 
                A[k, k - Nx] = diff * ( 1. / dx ** 2) 
                u[k, 0] = np.exp(-10. * (((x[m]-0.5)**2) + ((y[n]-0.5)**2)))

    if (method == "FE"):
        # Solve with Forward Euler
        for k in range(Nt - 1):
            u[1:-1, (k + 1):(k + 2)] = u[1:-1, k:(k + 1)] + dt * (f(t[k], u[1:-1, k:(k + 1)]))

    elif (method == "TRAP"):
        I = np.eye(N_total - 2)
        A_lhs = (I - (dt / 2) * A[1:-1, 1:-1])
        A_rhs = (I + (dt / 2) * A[1:-1, 1:-1])
        for k in range(Nt - 1):
            u[1:-1, (k + 1):(k + 2)] = np.linalg.solve(A_lhs, A_rhs @ u[1:-1, k:(k + 1)])

    X, Y = np.meshgrid(x, y)
    U = u[:, -1].reshape((Nx, Nx))

    # ax = plt.axes(projection='3d')
    # ax.plot_surface(X, Y, U)
    # plt.show()
    
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(2, 2, 1, projection='3d')
    # ax.plot_surface(X, Y, u[:, 0].reshape((Nx, Nx)), cmap='viridis')
    # ax.set_title('t=0')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('U')

    # ax = fig.add_subplot(2, 2, 2, projection='3d')
    # ax.plot_surface(X, Y, u[:, Nt//3].reshape((Nx, Nx)), cmap='viridis')
    # ax.set_title(r't=$\frac{1}{3}$')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('U')

    # ax = fig.add_subplot(2, 2, 3, projection='3d')
    # ax.plot_surface(X, Y, u[:, 2 * Nt//3].reshape((Nx, Nx)), cmap='viridis')
    # ax.set_title(r't=$\frac{2}{3}$')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('U')

    # ax = fig.add_subplot(2, 2, 4, projection='3d')
    # ax.plot_surface(X, Y, u[:, -1].reshape((Nx, Nx)), cmap='viridis')
    # ax.set_title('t=1')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('U')

    # plt.tight_layout()
    # if (method == 'FE'):
    #     fig.suptitle('Forward Euler with Nt = ' + str(Nt), fontsize=16)
    # elif (method == 'TRAP'):
    #     fig.suptitle('Trapezoidal method with Nt = ' + str(Nt), fontsize=16)
    # plt.savefig('p3' + method + str(Nt) + '.pdf')
    # plt.clf() 

    return U[Nx//2, Nx//2]
#------------------------------------------------------------------------------- 

'''P1'''
A1, A2 = p1(257, "FE")
A3, A4 = p1(139, "FE")
A5, A6 = p1(6, "BE")
A7, A8 = p1(51, "BE")

'''P2'''
A9, A10 = p2(56, "FE")
A11, A12 = p2(201, "FE")
A13, A14 = p2(11, "TRAP")
A15, A16 = p2(101, "TRAP")

'''P3'''
A17 =  p3(4, "FE")
A18 =  p3(101, "FE")
A19 =  p3(4, "TRAP")
A20 =  p3(101, "TRAP")
#------------------------------------------------------------------------------- 

# print("A1 = {}".format(A1))
# print("A2 = {}".format(A2))
# print("A3 = {}".format(A3))
# print("A4 = {}".format(A4))
# print("A5 = {}".format(A5))
# print("A6 = {}".format(A6))
# print("A7 = {}".format(A7))
# print("A8 = {}".format(A8))
# print("A9 = {}".format(A9))
# print("A10 = {}".format(A10))
# print("A11 = {}".format(A11))
# print("A12 = {}".format(A12))
# print("A13 = {}".format(A13))
# print("A14 = {}".format(A14))
# print("A15 = {}".format(A15))
# print("A16 = {}".format(A16))
# print("A17 = {}".format(A17))
# print("A18 = {}".format(A18))
# print("A19 = {}".format(A19))
# print("A20 = {}".format(A20))
#------------------------------------------------------------------------------- 