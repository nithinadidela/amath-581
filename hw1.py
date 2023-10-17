import numpy as np
import matplotlib.pyplot as plt

# Problem 1
# 1a
def p1_f(t_t, t_x):
    return -4. * t_x * np.sin(t_t)

# Forward Euler 
def p1_forward_euler(p1_f, t_x0, t_dt, t_time):
    x = np.zeros_like(t_time)
    x[0] = t_x0
    for i in range(len(t_time)-1):
       x[i+1] = x[i] + p1_f(t_time[i], x[i]) * t_dt 
    return x

# 1a1
p1a1_dt = 2. ** (-5)
p1a1_time = np.arange(0, 8 + 0.5 * p1a1_dt, p1a1_dt)
p1a1_x0 = 1.
p1a1_ex = np.exp(4. * (np.cos(p1a1_time)- 1.))
p1a1_x = p1_forward_euler(p1_f, p1a1_x0, p1a1_dt, p1a1_time)
p1a1_LTE = np.abs(p1a1_x[1] - p1a1_ex[1])
p1a1_GE = np.abs(p1a1_x[-1] - p1a1_ex[-1])

# plt.plot(p1a1_time, p1a1_ex, 'r-')
# plt.plot(p1a1_time, p1a1_x , 'b--')
# plt.show()

A1 = p1a1_LTE
print("A1", A1)
A2 = p1a1_GE
print("A2", A2)
#------------------------------------------------------------------------------- 

# 1a2
p1a2_dt = 2. ** (-6)
p1a2_time = np.arange(0, 8 + 0.5 * p1a2_dt, p1a2_dt)
p1a2_x0 = 1.
p1a2_ex = np.exp(4. * (np.cos(p1a2_time)- 1.))
p1a2_x = p1_forward_euler(p1_f, p1a2_x0, p1a2_dt, p1a2_time)
p1a2_LTE = np.abs(p1a2_x[1] - p1a2_ex[1])
p1a2_GE = np.abs(p1a2_x[-1] - p1a2_ex[-1])

# plt.plot(p1a2_time, p1a2_ex, 'r-')
# plt.plot(p1a2_time, p1a2_x , 'b--')
# plt.show()

A3 = p1a2_LTE
print("A3", A3)
A4 = p1a2_GE
print("A4", A4)
#------------------------------------------------------------------------------- 
#------------------------------------------------------------------------------- 

# 1b
# Heun's method
def p1_Heun(p1_f, t_x0, t_dt, t_time):
    x = np.zeros_like(t_time)
    x[0] = t_x0
    for i in range(len(t_time)-1):
       x[i+1] = x[i] + 0.5 * t_dt * (p1_f(t_time[i], x[i]) + p1_f(t_time[i] + t_dt, x[i] + t_dt * p1_f(t_time[i], x[i])))
    return x

# 1b1
p1b1_dt = 2. ** (-5)
p1b1_time = np.arange(0, 8 + 0.5 * p1b1_dt, p1b1_dt)
p1b1_x0 = 1.
p1b1_ex = np.exp(4. * (np.cos(p1b1_time)- 1.))
p1b1_x = p1_Heun(p1_f, p1b1_x0, p1b1_dt, p1b1_time)
p1b1_LTE = np.abs(p1b1_x[1] - p1b1_ex[1])
p1b1_GE = np.abs(p1b1_x[-1] - p1b1_ex[-1])

# plt.plot(p1b1_time, p1b1_ex, 'r-')
# plt.plot(p1b1_time, p1b1_x , 'b--')
# plt.show()

A5 = p1b1_LTE
print("A5", A5)
A6 = p1b1_GE
print("A6", A6)
#------------------------------------------------------------------------------- 

# 1b2
p1b2_dt = 2. ** (-6)
p1b2_time = np.arange(0, 8 + 0.5 * p1b2_dt, p1b2_dt)
p1b2_x0 = 1.
p1b2_ex = np.exp(4. * (np.cos(p1b2_time)- 1.))
p1b2_x = p1_Heun(p1_f, p1b2_x0, p1b2_dt, p1b2_time)
p1b2_LTE = np.abs(p1b2_x[1] - p1b2_ex[1])
p1b2_GE = np.abs(p1b2_x[-1] - p1b2_ex[-1])

# plt.plot(p1b2_time, p1b2_ex, 'r-')
# plt.plot(p1b2_time, p1b2_x , 'b--')
# plt.show()

A7 = p1b2_LTE
print("A7", A7)
A8 = p1b2_GE
print("A8", A8)
#------------------------------------------------------------------------------- 
#------------------------------------------------------------------------------- 

# 1c
# RK2 method
def p1_RK2(p1_f, t_x0, t_dt, t_time):
    x = np.zeros_like(t_time)
    x[0] = t_x0
    for i in range(len(t_time)-1):
       x[i+1] = x[i] + 1. * t_dt * p1_f(t_time[i] + 0.5 * t_dt, x[i] + 0.5 * t_dt * p1_f(t_time[i], x[i]))
    return x

# 1c1
p1c1_dt = 2. ** (-5)
p1c1_time = np.arange(0, 8 + 0.5 * p1c1_dt, p1c1_dt)
p1c1_x0 = 1.
p1c1_ex = np.exp(4. * (np.cos(p1c1_time)- 1.))
p1c1_x = p1_RK2(p1_f, p1c1_x0, p1c1_dt, p1c1_time)
p1c1_LTE = np.abs(p1c1_x[1] - p1c1_ex[1])
p1c1_GE = np.abs(p1c1_x[-1] - p1c1_ex[-1])

# plt.plot(p1c1_time, p1c1_ex, 'r-')
# plt.plot(p1c1_time, p1c1_x , 'b--')
# plt.show()

A9 = p1c1_LTE
print("A9", A9)
A10 = p1c1_GE
print("A10", A10)
#------------------------------------------------------------------------------- 

# 1c2
p1c2_dt = 2. ** (-6)
p1c2_time = np.arange(0, 8 + 0.5 * p1c2_dt, p1c2_dt)
p1c2_x0 = 1.
p1c2_ex = np.exp(4. * (np.cos(p1c2_time)- 1.))
p1c2_x = p1_RK2(p1_f, p1c2_x0, p1c2_dt, p1c2_time)
p1c2_LTE = np.abs(p1c2_x[1] - p1c2_ex[1])
p1c2_GE = np.abs(p1c2_x[-1] - p1c2_ex[-1])

# plt.plot(p1c2_time, p1c2_ex, 'r-')
# plt.plot(p1c2_time, p1c2_x , 'b--')
# plt.show()

A11 = p1c2_LTE
print("A11", A11)
A12 = p1c2_GE
print("A12", A12)
#------------------------------------------------------------------------------- 
#------------------------------------------------------------------------------- 

# Problem 2
# 2a
def p2_f(t_t, t_x):
    return 8. * np.sin(t_x)

# predictor-corrector method 
def p2_predictor_corrector(p2_f, t_x0, t_dt, t_time):
    x = np.zeros_like(t_time)
    x[0] = t_x0
    x[1] = x[0] + t_dt * p2_f(t_time[0] + 0.5 * t_dt, x[0] + 0.5 * t_dt * p2_f(t_time[0], x[0]))
    for i in range(1, len(t_time)-1):
        xp =  x[i] + 0.5 * t_dt * (3. * p2_f(t_time[i], x[i]) - p2_f(t_time[i-1], x[i-1]))
        x[i+1] = x[i] + 0.5 * t_dt * p2_f(t_time[i], xp) + p2_f(t_time[i], x[i])
    return x

# 2a
p2a_dt = 0.1  
p2a_time = np.arange(0, 2 + 0.5 * p2a_dt, p2a_dt)
p2a_x0 = 1.
p2a_ex = 2. * np.arctan((np.exp(8. * p2a_time)))
p2a_x = p2_predictor_corrector(p2_f, p2a_x0, p2a_dt, p2a_time)
p2a_LTE = np.abs(p2a_x[1] - p2a_ex[1])
p2a_GE = np.abs(p2a_x[-1] - p2a_ex[-1])

plt.plot(p2a_time, p2a_ex, 'r-')
plt.plot(p2a_time, p2a_x , 'b--')
plt.show()

A13 = p2a_x[-1]
print("A13", A13)
A14 = p2a_GE
print("A14", A14)
#-------------------------------------------------------------------------------