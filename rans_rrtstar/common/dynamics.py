from dataclasses import dataclass

import numpy as np
from scipy import signal
import casadi as cas
from casadi import SX, Function

from rans_rrtstar.config import DT, STEER_TIME, VELMIN, VELMAX, ANGVELMIN, ANGVELMAX, ENVAREA


@dataclass
class DynamicsData:
    num_states: int
    num_controls: int


class Dynamics:
    def __init__(self, n=1, m=1, Ts=DT):
        self.n = n
        self.m = m
        self.Ts = Ts

    def ctime_dynamics(self, x, u):
        raise NotImplementedError()

    def ctime_jacobian(self, x, u):
        raise NotImplementedError()

    def dtime_dynamics(self, x, u, Ts=None, method='euler'):
        # Discrete-time nonlinear dynamics
        if Ts is None:
            Ts = self.Ts
        # Euler method
        if method == 'euler':
            return x + self.ctime_dynamics(x, u)*Ts
        else:
            raise ValueError("Only the 'euler' method is a valid choice.")

    def dtime_jacobian(self, x, u, Ts=None, method='euler'):
        # Linearized discrete-time dynamics
        n, m = x.size, u.size
        A, B = self.ctime_jacobian(x, u)
        if Ts is None:
            Ts = self.Ts
        if method == 'euler':
            Ad = np.eye(n) + A*Ts
            Bd = B*Ts
        else:
            C, D = np.eye(n), np.zeros([n, m])
            sysd = signal.cont2discrete((A, B, C, D), Ts, method)
            return sysd[0], sysd[1]
        return Ad, Bd

    def one_step_sim_diff(self, x_now, u_now, x_next):
        x_next_sim = self.dtime_dynamics(x_now, u_now)
        diff = np.abs(x_next - x_next_sim)
        return diff

    def check_entire_traj(self, x, u, tol=1e-14):
        failures = []
        failure_times = []
        for ii in range(len(u) - 1):
            x_now = x[ii]
            u_now = u[ii]
            x_next = x[ii + 1]
            diff = self.one_step_sim_diff(x_now, u_now, x_next)
            if np.sum(diff) > tol:
                failures.append(diff)
                failure_times.append(ii)
        return [failures, failure_times]

    def make_dynamics_dataclass(self):
        return DynamicsData(self.n, self.m)


class UnicycleDynamics(Dynamics):
    def __init__(self, Ts=DT):
        super().__init__(n=3, m=2, Ts=Ts)

    def ctime_dynamics(self, x, u, fset='np'):
        # Continuous-time nonlinear dynamics
        # Choose the function set to use, NumPy or CasADi
        if fset == 'np':
            cos, sin = np.cos, np.sin
            horzcat = lambda *args: np.array(args)
        elif fset == 'cas':
            cos, sin = cas.cos, cas.sin
            horzcat = cas.horzcat
        else:
            raise ValueError('Invalid fset!')
        return horzcat(u[0]*cos(x[2]), u[0]*sin(x[2]), u[1])

    def ctime_jacobian(self, x, u):
        # Linearized continuous-time dynamics
        A = np.array([[0, 0, -u[0]*np.sin(x[2])],
                      [0, 0, u[0]*np.cos(x[2])],
                      [0, 0, 0]])
        B = np.array([[np.cos(x[2]), 0],
                      [np.sin(x[2]), 0],
                      [0, 1]])
        return A, B

    def ctime_dynamics_cas(self):
        # Define symbolic states using Casadi SX
        x = SX.sym('x')  # x position
        y = SX.sym('y')  # y position
        theta = SX.sym('theta')  # heading angle
        states = cas.vertcat(x, y, theta)  # all three states

        # Define symbolic inputs using Casadi SX
        v = SX.sym('v')  # linear velocity
        omega = SX.sym('omega')  # angular velocity
        controls = cas.vertcat(v, omega)  # both controls

        # RHS of nonlinear unicycle dynamics (continuous time model)
        # rhs = cas.horzcat(v*cas.cos(theta), v*cas.sin(theta), omega)
        rhs = self.ctime_dynamics(states, controls, fset='cas')

        # Nonlinear State Update function f(x,u)
        # Given {states, controls} as inputs, returns {rhs} as output
        f = Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])
        return f

    def state_control_constraints(self):
        N = STEER_TIME  # Prediction horizon

        x_min, x_max, y_min, y_max = ENVAREA  # min, max positions in x, y directions
        theta_min, theta_max = -np.inf, np.inf  # min, max angular positions

        lbx = []
        ubx = []
        lbg = 0.0
        ubg = 0.0
        # Upper and lower bounds on controls
        for _ in range(N):
            lbx.append(VELMIN)
            ubx.append(VELMAX)
        for _ in range(N):
            lbx.append(ANGVELMIN)
            ubx.append(ANGVELMAX)
        # Upper and lower bounds on states
        for _ in range(N+1):
            lbx.append(x_min)
            ubx.append(x_max)
        for _ in range(N+1):
            lbx.append(y_min)
            ubx.append(y_max)
        for _ in range(N+1):
            lbx.append(theta_min)
            ubx.append(theta_max)

        # Create the arguments dictionary to hold the constraint values
        constraint_argdict = {'lbg': lbg,
                              'ubg': ubg,
                              'lbx': lbx,
                              'ubx': ubx}
        return constraint_argdict


DYN = UnicycleDynamics()
