import numpy as np
import config as cf
Ts = cf.DT

# Continuous-time nonlinear dynamics
def ctime_dynamics(x, u):
    return np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1]])


# Discrete-time nonlinear dynamics
def dtime_dynamics(x, u, Ts):
    # Euler method
    return x + ctime_dynamics(x, u)*Ts

def one_step_sim(x_now, u_now, x_next):
    x_next_sim = dtime_dynamics(x_now, u_now, Ts)
    diff = np.abs(x_next - x_next_sim)
    # print(diff)
    return diff

def check_entire_traj(x,u,tol=1e-14):
    failures = []
    failure_times = []
    for ii in range(len(u) - 1):
        x_now = x[ii]
        u_now = u[ii]
        x_next = x[ii + 1]
        diff = one_step_sim(x_now, u_now, x_next)
        if np.sum(diff) > tol:
            print(diff)
            failures.append(diff)
            failure_times.append(ii)
    return [failures, failure_times]