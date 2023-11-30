import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy

def p1():
    # y'' + 0.1 y' + sin(y) = 0
    x0 = 0
    xN = 6
    y0 = 0.5
    yN = 0.5

    dx = 0.06
    x = np.arange(x0, xN + 0.5 * dx, dx)
    N = len(x)
    den = (1. / (dx**2)) + (0.1 / (2. * dx))

    def F(y):
        z = np.zeros((N, 1))
        z[0] = y[0] - y0
        z[-1] = y[-1] - yN
        for k in range(1, N - 1):
            z[k] =  y[k - 1] * (1. - 0.05 * dx)\
                    - 2 * y[k] + (np.sin(y[k]) * (dx**2))\
                    + y[k + 1] * (1. + 0.05 * dx)
        return z 

    def jacobian(y):
        J = np.zeros((N, N))
        J[0, 0] = 1
        J[-1, -1] = 1
        for k in range(1, N-1):
            J[k, k - 1] = (1. - 0.05 * dx)
            J[k, k] = -2 + (np.cos(y[k]) * (dx**2))
            J[k, k + 1] = (1. + 0.05 * dx) 
        return J

    """P1a"""
    y = np.ones((N, 1)) * 0.5
    k = 0
    max_steps = 500
    while np.max(np.abs(F(y))) >= 1e-8 and k < max_steps:
        change_in_y = np.linalg.solve(jacobian(y), F(y))
        y = y - change_in_y
        k = k + 1

    y = y.reshape(N)
    # print("k = {}".format(k))

    # plt.plot(x, y, 'k', x0, y0, 'ro', xN, yN, 'ro')
    # plt.show()

    l_A1 = y[N//2]
    l_A2 = np.max(y)
    l_A3 = np.min(y)

    """P1b"""
    y = np.zeros((N, 1))
    for i in range(0, N):
        t = x[i]
        y[i] =      (0.005 * t**4) \
                    - (0.07 * t**3) \
                    + (0.66 * t**2) \
                    - (2.56 * t) + 0.55
    k = 0
    max_steps = 500
    while np.max(np.abs(F(y))) >= 1e-8 and k < max_steps:
        change_in_y = np.linalg.solve(jacobian(y), F(y))
        y = y - change_in_y
        k = k + 1

    y = y.reshape(N)
    # print("k = {}".format(k))

    # plt.plot(x, y, 'k', x0, y0, 'ro', xN, yN, 'ro')
    # plt.show()

    l_A4 = y[N//2]
    l_A5 = np.max(y)
    l_A6 = np.min(y)

    return l_A1, l_A2, l_A3, l_A4, l_A5, l_A6
#------------------------------------------------------------------------------- 

def p2a():
    # u_xx + u_yy = 0
    # u(x, 0) = x^2 - 3x
    # u(x, 3) = sin(2 pi x / 3) 
    # u(0, y) = sin(pi y / 3)
    # u(3, y) = 3y - y^2 

    x0 = 0
    xN = 3
    y0 = 0
    yN = 3

    dx = 0.05
    x = np.arange(x0, xN + 0.5 * dx, dx)
    y = np.arange(y0, yN + 0.5 * dx, dx)
    N = len(x)
    
    # print("N = {}".format(N))
    # print("dx = {}".format(dx))

    # Boundary conditions
    # u(x, 0)
    def down(x):
        return x**2 - 3. * x

    # u(x, 3)
    def up(x):
        return np.sin(2. * np.pi * x / 3.)

    # u(0, y)
    def left(y):
        return np.sin(np.pi * y / 3.) 

    # u(3, y)
    def right(y):
        return 3. * y - y**2 

    # Solve Au = b
    N_total = N * N
    # print("N_total = {}".format(N_total))
    # print("Entries in matrix = {}".format(N_total ** 2))
    A = np.zeros((N_total, N_total))
    b = np.zeros((N_total, 1))

    def point2ind(m, n):
        return n * N + m

    for n in range(N):
        for m in range(N):
            k = point2ind(m, n)
            if n == 0:
                A[k, k] = 1
                b[k] = down(x[m])
            elif n == N-1:
                A[k, k] = 1
                b[k] = up(x[m])
            elif m == 0:
                A[k, k] = 1
                b[k] = left(y[n])
            elif m == N-1:
                A[k, k] = 1
                b[k] = right(y[n])
            else:
                A[k, k] = -4 / dx ** 2
                A[k, k + 1] = 1 / dx ** 2
                A[k, k - 1] = 1 / dx ** 2
                A[k, k + N] = 1 / dx ** 2
                A[k, k - N] = 1 / dx ** 2
                b[k] = 0 # f(x[m], y[n])

    u = np.linalg.solve(A, b)
    U = u.reshape((N, N))
    l_A7 = U[N//3, N//3]
    l_A8 = U[2*N//3, 2*N//3]

    return l_A7, l_A8
#------------------------------------------------------------------------------- 

def p2b():
    # u_xx + u_yy = 0
    # u(x, 0) = x^2 - 3x
    # u(x, 3) = sin(2 pi x / 3) 
    # u(0, y) = sin(pi y / 3)
    # u(3, y) = 3y - y^2 

    x0 = 0
    xN = 3
    y0 = 0
    yN = 3

    dx = 0.015
    x = np.arange(x0, xN + 0.5 * dx, dx)
    y = np.arange(y0, yN + 0.5 * dx, dx)
    N = len(x)
    
    # print("N = {}".format(N))
    # print("dx = {}".format(dx))

    # Boundary conditions
    # u(x, 0)
    def down(x):
        return x**2 - 3. * x

    # u(x, 3)
    def up(x):
        return np.sin(2. * np.pi * x / 3.)

    # u(0, y)
    def left(y):
        return np.sin(np.pi * y / 3.) 

    # u(3, y)
    def right(y):
        return 3. * y - y**2 

    # Solve Au = b
    N_total = (N - 2) * (N - 2)
    # print("N_total = {}".format(N_total))
    # print("Entries in matrix = {}".format(N_total ** 2))
    A = scipy.sparse.dok_array((N_total, N_total))
    b = np.zeros((N_total, 1))

    def point2ind(m, n):
        return (n - 1) * (N - 2) + m - 1

    for n in range(1, N-1):
        for m in range(1, N-1):
            k = point2ind(m, n)
            A[k, k] = -4 / dx ** 2
            if m > 1:
                A[k, k - 1] = 1 / dx ** 2
            else:
                b[k] = b[k] - left((y[n])) / dx ** 2 
            if n < N - 2:
                A[k, k + N - 2] = 1 / dx ** 2
            else:
                b[k] = b[k] - up(x[m]) / dx ** 2
            if m < N - 2:
                A[k, k + 1] = 1 / dx ** 2
            else:
                b[k] = b[k] - right(y[n]) / dx ** 2
            if n > 1:
                A[k, k - (N - 2)] = 1 / dx ** 2
            else:
                b[k] = b[k] - down(x[m])/ dx ** 2
    A = A.tocsc()
    

    # Solve system
    u = scipy.sparse.linalg.spsolve(A, b)

    U_int = u.reshape((N-2, N-2))
    U = np.zeros((N, N))
    U[1:(N-1), 1:(N-1)] = U_int
    U[0, :] = down(x)
    U[N-1, :] = up(x)
    U[:, 0] = left(y)
    U[:, N-1] = right(y)

    l_A9 = U[N//3, N//3]
    l_A10 = U[2*N//3 - 1, 2*N//3 - 1]
    return l_A9, l_A10
#------------------------------------------------------------------------------- 

def p3(dx, dy):
    # u_xx + u_yy = -e^(-2(x^2 + y^2))
    # u(x, -1) = 0 
    # u(x, 1) = (x^3 - x) / 3
    # u(-1, y) = 0 
    # u(1, y) = 0 
    x0 = -1 
    xN = 1
    y0 = -1
    yN = 1

    x = np.arange(x0, xN + 0.5 * dx, dx)
    y = np.arange(y0, yN + 0.5 * dy, dy)
    Nx = len(x)
    Ny = len(y)
    
    # Boundary conditions
    # u(x, 0)
    def down(x):
        return 0 

    # u(x, 3)
    def up(x):
        return (x**3 - x) / 3.

    # u(0, y)
    def left(y):
        return 0 

    # u(3, y)
    def right(y):
        return 0 

    # Solve Au = b
    N_total = (Nx - 2) * (Ny - 2)
    A = scipy.sparse.dok_array((N_total, N_total))
    b = np.zeros((N_total, 1))

    def point2ind(m, n):
        return (n - 1) * (Nx - 2) + m - 1

    for n in range(1, Ny-1):
        for m in range(1, Nx-1):
            k = point2ind(m, n)

            A[k, k] = -2 *((1. / dx ** 2) + (1. / dy ** 2)) 
            b[k] = -1. * np.exp(-2. * (x[m]**2 + y[n]**2)) 

            if m > 1:
                A[k, k - 1] = 1 / dx ** 2
            else:
                b[k] = b[k] - left((y[n])) / dx ** 2 

            if n < Ny - 2:
                A[k, k + (Nx - 2)] = 1 / dy ** 2
            else:
                b[k] = b[k] - up(x[m]) / dy ** 2

            if m < Nx - 2:
                A[k, k + 1] = 1 / dx ** 2
            else:
                b[k] = b[k] - right(y[n]) / dx ** 2

            if n > 1:
                A[k, k - (Nx - 2)] = 1 / dy ** 2
            else:
                b[k] = b[k] - down(x[m])/ dy ** 2

    A = A.tocsc()

    # Solve system
    u = scipy.sparse.linalg.spsolve(A, b)

    U_int = u.reshape((Ny-2, Nx-2))
    U = np.zeros((Ny, Nx))
    U[1:(Ny-1), 1:(Nx-1)] = U_int

    U[0, :] = down(x)
    U[Ny-1, :] = up(x)
    U[:, 0] = left(y)
    U[:, Nx-1] = right(y)

    # print(x[Nx//2], y[Ny//2])
    # print(x[Nx//4], y[3 * Ny//4])
    l_A1 = U[Ny//2, Nx//2]
    l_A2 = U[3*Ny//4, Nx//4]
    return l_A1, l_A2
#------------------------------------------------------------------------------- 

def p4a(dx):
    # u_xx + u_yy = - sin(pi * x) sin(pi * y)
    # u(x, 0) = 0
    # u(x, 2) = 0
    # u(0, y) = 0
    # u(2, y) = 0

    x0 = 0
    xN = 2
    y0 = 0
    yN = 2

    x = np.arange(x0, xN + 0.5 * dx, dx)
    y = np.arange(y0, yN + 0.5 * dx, dx)
    N = len(x)
    
    # Boundary conditions
    # u(x, 0)
    def down(x):
        return 0 

    # u(x, 3)
    def up(x):
        return 0 

    # u(0, y)
    def left(y):
        return 0 

    # u(3, y)
    def right(y):
        return 0 

    # Solve Au = b
    N_total = (N - 2) * (N - 2)
    # print("N_total = {}".format(N_total))
    # print("Entries in matrix = {}".format(N_total ** 2))
    A = scipy.sparse.dok_array((N_total, N_total))
    b = np.zeros((N_total, 1))

    def point2ind(m, n):
        return (n - 1) * (N - 2) + m - 1

    for n in range(1, N-1):
        for m in range(1, N-1):
            k = point2ind(m, n)
            A[k, k] = -4 / dx ** 2
            b[k] = -1. * np.sin(np.pi * x[m]) * np.sin(np.pi * y[n])

            if m > 1:
                A[k, k - 1] = 1 / dx ** 2
            else:
                b[k] = b[k] - left((y[n])) / dx ** 2 

            if n < N - 2:
                A[k, k + N - 2] = 1 / dx ** 2
            else:
                b[k] = b[k] - up(x[m]) / dx ** 2

            if m < N - 2:
                A[k, k + 1] = 1 / dx ** 2
            else:
                b[k] = b[k] - right(y[n]) / dx ** 2

            if n > 1:
                A[k, k - (N - 2)] = 1 / dx ** 2
            else:
                b[k] = b[k] - down(x[m])/ dx ** 2

    A = A.tocsc()
    
    # Solve system
    u = scipy.sparse.linalg.spsolve(A, b)

    U_int = u.reshape((N-2, N-2))
    U = np.zeros((N, N))
    U[1:(N-1), 1:(N-1)] = U_int
    U[0, :] = down(x)
    U[N-1, :] = up(x)
    U[:, 0] = left(y)
    U[:, N-1] = right(y)

    U_true_sol = np.zeros((N, N))
    for n in range(1, N-1):
        for m in range(1, N-1):
            U_true_sol[m, n] = np.sin(np.pi * x[m]) * np.sin(np.pi * y[n]) \
                / (2. * np.pi**2) 
    
    error = np.max(np.abs(U_true_sol - U))

    if (dx == 2**(-5)): 
      X, Y = np.meshgrid(x, y)
      plt.contourf(X, Y, U_true_sol)
      cbar = plt.colorbar()
      plt.axis('equal')
      plt.savefig('amath581_hw3_p4a_U.pdf')
      plt.clf()
    return error
#------------------------------------------------------------------------------- 

def p4b_sparse(dx):
    # u_xx + u_yy = - sin(pi * x) sin(pi * y)
    # u(x, 0) = 0
    # u(x, 2) = 0
    # u(0, y) = 0
    # u(2, y) = 0

    x0 = 0
    xN = 2
    y0 = 0
    yN = 2

    x = np.arange(x0, xN + 0.5 * dx, dx)
    y = np.arange(y0, yN + 0.5 * dx, dx)
    N = len(x)
    
    # Boundary conditions
    # u(x, 0)
    def down(x):
        return 0 

    # u(x, 3)
    def up(x):
        return 0 

    # u(0, y)
    def left(y):
        return 0 

    # u(3, y)
    def right(y):
        return 0 

    # Solve Au = b
    N_total = (N - 2) * (N - 2)
    # print("N_total = {}".format(N_total))
    # print("Entries in matrix = {}".format(N_total ** 2))
    A = scipy.sparse.dok_array((N_total, N_total))
    b = np.zeros((N_total, 1))

    def point2ind(m, n):
        return (n - 1) * (N - 2) + m - 1

    neighbours = np.zeros((N_total, N_total))
    for n in range(1, N-1):
        for m in range(1, N-1):
            k = point2ind(m, n)
            A[k, k] = -20. / (6. * (dx ** 2))
            b[k] = -1. * np.sin(np.pi * x[m]) * np.sin(np.pi * y[n])

            if m > 1:
                A[k, k - 1] = 4. / (6. * (dx ** 2))
            else:
                b[k] = b[k] - left((y[n])) / dx ** 2 

            if n < N - 2:
                A[k, k + N - 2] = 4. / (6. * (dx ** 2)) 
            else:
                b[k] = b[k] - up(x[m]) / dx ** 2

            if m < N - 2:
                A[k, k + 1] = 4. / (6. * (dx ** 2)) 
            else:
                b[k] = b[k] - right(y[n]) / dx ** 2

            if n > 1:
                A[k, k - (N - 2)] = 4. / (6. * (dx ** 2)) 
            else:
                b[k] = b[k] - down(x[m])/ dx ** 2

            if (m > 1) and (n > 1):
                A[k, (k-1) - (N-2)] = 1. / (6. * (dx ** 2))  

            if (m > 1) and (n < (N - 2)):
                A [k, (k-1) + (N-2)] = 1. / (6. * (dx ** 2)) 

            if (m < (N-2)) and (n < (N - 2)):
                A [k, (k+1) + (N-2)] = 1. / (6. * (dx ** 2)) 

            if (m < (N-2)) and (n > 1):
                A [k, (k+1) - (N-2)] = 1. / (6. * (dx ** 2)) 

    A = A.tocsc()
    
    # Solve system
    u = scipy.sparse.linalg.spsolve(A, b)

    U_int = u.reshape((N-2, N-2))
    U = np.zeros((N, N))
    U[1:(N-1), 1:(N-1)] = U_int
    U[0, :] = down(x)
    U[N-1, :] = up(x)
    U[:, 0] = left(y)
    U[:, N-1] = right(y)

    U_true_sol = np.zeros((N, N))
    for n in range(1, N-1):
        for m in range(1, N-1):
            U_true_sol[m, n] = np.sin(np.pi * x[m]) * np.sin(np.pi * y[n]) \
                / (2. * np.pi**2) 
    
    error = np.max(np.abs(U_true_sol - U))

    if (dx == 2**(-5)): 
      X, Y = np.meshgrid(x, y)
      plt.contourf(X, Y, U_true_sol)
      cbar = plt.colorbar()
      plt.axis('equal')
      plt.savefig('amath581_hw3_p4b_U.pdf')
      plt.clf()
    return error
#------------------------------------------------------------------------------- 

def p4b(dx):
    # u_xx + u_yy = 0
    # u(x, 0) = x^2 - 3x
    # u(x, 3) = sin(2 pi x / 3) 
    # u(0, y) = sin(pi y / 3)
    # u(3, y) = 3y - y^2 

    x0 = 0
    xN = 2
    y0 = 0
    yN = 2

    x = np.arange(x0, xN + 0.5 * dx, dx)
    y = np.arange(y0, yN + 0.5 * dx, dx)
    N = len(x)
    
    # Solve Au = b
    N_total = N * N
    # print("N_total = {}".format(N_total))
    # print("Entries in matrix = {}".format(N_total ** 2))
    A = np.zeros((N_total, N_total))
    b = np.zeros((N_total, 1))

    def point2ind(m, n):
        return n * N + m

    for n in range(N):
        for m in range(N):
            k = point2ind(m, n)
            if n == 0:
                A[k, k] = 1
            elif n == N-1:
                A[k, k] = 1
            elif m == 0:
                A[k, k] = 1
            elif m == N-1:
                A[k, k] = 1
            else:
                A[k, k] = -20. / (6. * dx ** 2)
                A[k, k + 1] = 4. / (6. * dx ** 2)
                A[k, k - 1] = 4. / (6. * dx ** 2)
                A[k, k + N] = 4. / (6. * dx ** 2)
                A[k, k - N] = 4. / (6. * dx ** 2)
                A[k, k + 1 - N] = 1. / (6. * (dx ** 2))    
                A[k, k + 1 + N] = 1. / (6. * (dx ** 2)) 
                A[k, k - 1 - N] = 1. / (6. * (dx ** 2)) 
                A[k, k - 1 + N] = 1. / (6. * (dx ** 2)) 
                b[k] = -1. * np.sin(np.pi * x[m]) * np.sin(np.pi * y[n])

    u = np.linalg.solve(A, b)
    U = u.reshape((N, N))
    U_true_sol = np.zeros((N, N))
    for n in range(1, N-1):
        for m in range(1, N-1):
            U_true_sol[m, n] = np.sin(np.pi * x[m]) * np.sin(np.pi * y[n]) \
                / (2. * np.pi**2) 
    
    error = np.max(np.abs(U_true_sol - U))
    return error
#------------------------------------------------------------------------------- 

def p4c(dx):
    x0 = 0
    xN = 2
    y0 = 0
    yN = 2

    x = np.arange(x0, xN + 0.5 * dx, dx)
    y = np.arange(y0, yN + 0.5 * dx, dx)
    N = len(x)

    f = np.zeros((N, N))
    for n in range(1, N-1):
        for m in range(1, N-1):
            f[m, n] = np.sin(np.pi * x[m]) * np.sin(np.pi * y[n]) \
                / (2. * np.pi**2)  

    
    delf = np.zeros((N, N))
    for n in range(1, N-1):
        for m in range(1, N-1):
            delf[m, n] = -1. * (np.sin(np.pi * x[m]) * np.sin(np.pi * y[n])) 

    
    lap9 = np.zeros((N, N))
    for n in range(1, N-1):
        for m in range(1, N-1):
           lap9[m, n] = (4. * (f[m-1, n] + f[m+1, n] + f[m, n-1] + f[m, n+1]) \
                         + 1. * (f[m-1, n-1] + f[m-1, n+1] + f[m+1, n-1] + f[m+1, n+1]) \
                         - 20. * (f[m, n])) / (6. * dx**2) 
    
    discretization_error = lap9 - f
    X, Y = np.meshgrid(x,y)
    ax1 = plt.axes(projection='3d')
    ax1.plot_surface(X, Y, discretization_error, cmap='viridis')
    ax1.set_title("Discretization Error of 9-point Laplacian")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("Discretization Error")
    plt.savefig("amath581_hw3_p4c_de.pdf")
    plt.clf

    ax2 = plt.axes(projection='3d')
    ax2.plot_surface(X, Y, delf, cmap='viridis')
    ax2.set_title("Analytical Laplacian")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("Analytical Laplacian of f")
    plt.savefig("amath581_hw3_p4c_as.pdf")
    plt.clf
    

#------------------------------------------------------------------------------- 
# dx_list = 2. ** np.arange(-5, 0, +1)

# error_a = np.zeros_like(dx_list)
# error_b = np.zeros_like(dx_list)
# error_b_sparse = np.zeros_like(dx_list)

# for i in (range(len(dx_list))):
#     dx = dx_list[i]
#     error_a[i] = p4a(dx)
#     error_b[i] = p4b(dx)
#     error_b_sparse[i] = p4b_sparse(dx)

# p4a = PdfPages( 'amath581_hw3_p4a_error.pdf')
# fig, ax = plt.subplots()
# ax.loglog(dx_list, error_a, 'r-x',label='5-point Laplacian', base=2)

# x_ticks = dx_list
# x_ticks_labels = np.arange(-5, 0, 1) 
# ax.set_xticks(x_ticks, x_ticks_labels)
# ax.set_xlabel(r'$log_{2}(\Delta x$)')
# ax.set_ylabel(r'$log_{2}(E_N$)')
# # ax.set_ylim(2*1e-5, 2*1e-1)
# # ax.set_xlim(0.0, 16.0)
# plt.tight_layout()
# ax.legend(loc='best')
# p4a.savefig()
# p4a.close()

# p4b = PdfPages( 'amath581_hw3_p4b_error.pdf')
# fig, ax = plt.subplots()
# ax.loglog(dx_list, error_b, 'b-x',label='9-point Laplacian', base=2)
# ax.loglog(dx_list, error_b_sparse, 'ro',label='9-point Laplacian using Sparse A', base=2)

# x_ticks = dx_list
# x_ticks_labels = np.arange(-5, 0, 1) 
# ax.set_xticks(x_ticks, x_ticks_labels)
# ax.set_xlabel(r'$log_{2}(\Delta x$)')
# ax.set_ylabel(r'$log_{2}(E_N$)')
# # ax.set_ylim(2*1e-5, 2*1e-1)
# # ax.set_xlim(0.0, 16.0)
# plt.tight_layout()
# ax.legend(loc='best')
# p4b.savefig()
# p4b.close()

p4c(0.05)

A1, A2, A3, A4, A5, A6 = p1()
A7, A8 = p2a()
A9, A10 = p2b()
A11, A12 = p3(0.1, 0.05)
A13, A14 = p3(0.01, 0.025)
print("A1", A1)
print("A2", A2)
print("A3", A3)
print("A4", A4)
print("A5", A5)
print("A6", A6)
print("A7", A7)
print("A8", A8)
print("A9", A9)
print("A10", A10)
print("A11", A11)
print("A12", A12)
print("A13", A13)
print("A14", A14)