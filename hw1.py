import numpy as np
import matplotlib.pyplot as plt

# Problem 1a
def p1_f(t_x, t_t):
    return -4. * t_x * np.sin(t_t)

# Forward Euler 
def p1_forward_euler(p1_f, t_x0, t_dt, t_time):
    x = np.zeros_like(t_time)
    x[0] = t_x0
    for i in range(len(t_time)-1):
       x[i+1] = x[i] + p1_f(x[i], t_time[i]) * t_dt 
    return x

# 1a1
p1a1_dt = 2. * 1e-5
p1a1_time = np.arange(0, 8 + 0.5 * p1a1_dt, p1a1_dt)
p1a1_x0 = 1.
p1a1_ex = np.exp(4. * (np.cos(p1a1_time)- 1.))
p1a1_x = p1_forward_euler(p1_f, p1a1_x0, p1a1_dt, p1a1_time)
p1a1_LTE = np.abs(p1a1_x[1] - p1a1_ex[1])
p1a1_GE = np.abs(p1a1_x[-1] - p1a1_x[-1])

# plt.plot(p1a1_time, p1a1_ex, 'r-')
# plt.plot(p1a1_time, p1a1_x , 'b-')
# plt.show()

A1 = p1a1_LTE
A2 = p1a1_GE

# 1a2
p1a2_dt = 2. * 1e-6
p1a2_time = np.arange(0, 8 + 0.5 * p1a2_dt, p1a2_dt)
p1a2_x0 = 1.
p1a2_ex = np.exp(4. * (np.cos(p1a2_time)- 1.))
p1a2_x = p1_forward_euler(p1_f, p1a2_x0, p1a2_dt, p1a2_time)
p1a2_LTE = np.abs(p1a2_x[1] - p1a2_ex[1])
p1a2_GE = np.abs(p1a2_x[-1] - p1a2_x[-1])

plt.plot(p1a2_time, p1a2_ex, 'r-')
plt.plot(p1a2_time, p1a2_x , 'b-')
plt.show()

A3 = p1a2_LTE
A4 = p1a2_GE
