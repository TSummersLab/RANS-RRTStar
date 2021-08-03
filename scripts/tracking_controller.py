#!/usr/bin/env python3
"""
Changelog:
New in version 1_0:
- Create script to run and test functions in `lqr_uni.py`

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author:
Ben Gravell
Email:
benjamin.gravell@utdallas.edu
Github:
@BenGravell

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script does the following:
- Creates lqr and lqrm controllers. Edit `use_robust_lqr` to switch between lqr and lqrm.

"""

import os

import numpy as np

from config import SAVEPATH, DT, QLL, RLL, QTLL
from scripts.dynamics import DYN
from lqr import lqr, lqrm

from utility.pickle_io import pickle_import


class OpenLoopController:
    def __init__(self, u_ref_hist):
        self.u_ref_hist = u_ref_hist

    def compute_input(self, x, t):
        u_ref = self.u_ref_hist[t]
        return u_ref


class LQRController:
    def __init__(self, K_hist, L_hist, e_hist, z_hist, x_ref_hist, u_ref_hist):
        self.K_hist = K_hist
        self.L_hist = L_hist
        self.e_hist = e_hist
        self.z_hist = z_hist
        self.x_ref_hist = x_ref_hist
        self.u_ref_hist = u_ref_hist

    def compute_input(self, x, t):
        K = self.K_hist[t]
        L = self.L_hist[t]
        e = self.e_hist[t]
        z = self.z_hist[t]
        x_ref = self.x_ref_hist[t]
        u_ref = self.u_ref_hist[t]
        dx = x - x_ref
        u = np.dot(K, dx) + np.dot(L, z) + e + u_ref
        return u


def load_ref_traj(input_file):
    inputs_string = "inputs"
    states_file = input_file.replace(inputs_string, "states")

    input_file = os.path.join(SAVEPATH, input_file)
    states_file = os.path.join(SAVEPATH, states_file)

    # load inputs and states
    ref_inputs = pickle_import(input_file)
    ref_states = pickle_import(states_file)
    return ref_states, ref_inputs


def create_lqrm_controller(x_ref_hist, u_ref_hist,
                           use_robust_lqr=True, delta_theta_max=1*(2*np.pi/360), use_adversary=False):
    # delta_theta_max is the maximum assumed angle deviation in radians for robust control design

    T, n = x_ref_hist.shape
    T, m = u_ref_hist.shape
    # Compute linearized dynamics matrices along the reference trajectory
    A_hist = np.zeros([T, n, n])
    B_hist = np.zeros([T, n, m])
    for t in range(T):
        A_hist[t], B_hist[t] = DYN.dtime_jacobian(x_ref_hist[t], u_ref_hist[t])

    E_hist = np.zeros([T, n, n])  # TODO - make this something more meaningful
    W_hist = np.zeros([T, n, n])  # TODO - make this something more meaningful

    # Construct multiplicative noises and additive adversary
    c = 3  # Number of adversary inputs
    C_hist = np.zeros([T, n, c])
    for t in range(T):
        if use_adversary:
            # Adversary can push robot around isotropically in xy plane position and twist the robot angle a little
            C_hist[t] = np.array([[0.4, 0.0, 0.0],
                                  [0.0, 0.4, 0.0],
                                  [0.0, 0.0, 0.1]])
        else:
            # No adversary
            C_hist[t] = np.zeros([n, c])

    num_alphas = 2
    num_betas = 2
    num_gammas = 2
    alpha_var_hist = np.zeros([T, num_alphas])
    beta_var_hist = np.zeros([T, num_betas])
    gamma_var_hist = np.zeros([T, num_gammas])

    Ai_hist = np.zeros([T, num_alphas, n, n])
    Bi_hist = np.zeros([T, num_betas, n, m])
    Ci_hist = np.zeros([T, num_gammas, n, c])
    for t in range(T):
        v = u_ref_hist[t, 0]
        sin_theta = np.sin(x_ref_hist[t, 2])
        cos_theta = np.cos(x_ref_hist[t, 2])

        sin_delta_theta_max = np.sin(delta_theta_max)
        cos_delta_theta_max = np.cos(delta_theta_max)

        alpha_var_hist[t, 0] = v*DT*sin_delta_theta_max
        alpha_var_hist[t, 1] = v*DT*(1 - cos_delta_theta_max)
        beta_var_hist[t, 0] = sin_delta_theta_max
        beta_var_hist[t, 1] = 1 - cos_delta_theta_max

        Ai_hist[t, 0] = np.array([[0, 0, -sin_theta],
                                  [0, 0, cos_theta],
                                  [0, 0, 0]])
        Ai_hist[t, 1] = np.array([[0, 0, -cos_theta],
                                  [0, 0, -sin_theta],
                                  [0, 0, 0]])
        Bi_hist[t, 0] = np.array([[-sin_theta,  0],
                                  [cos_theta,  0],
                                  [0,  0]])
        Bi_hist[t, 1] = np.array([[cos_theta,  0],
                                  [sin_theta,  0],
                                  [0,  0]])

    # Construct cost matrices
    # We use the same cost matrices for all time steps, including the final time
    Qorg = np.diag([0, 0, 0])    # Penalty on state being far from origin
    Qref = QLL
    QTref = QTLL
    Rorg = RLL
    Rref = np.diag([0, 0])       # Penalty on input deviating from reference (deviation control effort)
    # Vorg = (1/robust_scale)*600*np.diag([2, 2, 1])       # Penalty on additive adversary
    Vorg = 1000 * np.diag([1, 1, 1])  # Penalty on additive adversary

    G_hist = np.zeros([T, n+m+c+n+m, n+m+c+n+m])
    for t in range(T):
        Znm, Zmn = np.zeros([n, m]), np.zeros([m, n])
        Znc, Zmc = np.zeros([n, c]), np.zeros([m, c])
        Zcn, Zcm = np.zeros([c, n]), np.zeros([c, m])

        G_hist[t] = np.block([[Qref+Qorg, Znm, Znc, Qorg, Znm],
                              [Zmn, Rref+Rorg, Zmc, Zmn, Rorg],
                              [Zcn, Zcm, -Vorg, Zcn, Zcm],
                              [Qorg, Znm, Znc, Qorg, Znm],
                              [Zmn, Rorg, Zmc, Zmn, Rorg]])

    # Terminal penalty
    G_hist[-1] = np.block([[QTref+Qorg, Znm, Znc, Qorg, Znm],
                           [Zmn, Rref+Rorg, Zmc, Zmn, Rorg],
                           [Zcn, Zcm, -Vorg, Zcn, Zcm],
                           [Qorg, Znm, Znc, Qorg, Znm],
                           [Zmn, Rorg, Zmc, Zmn, Rorg]])

    # Construct the exogenous signal
    z_hist = np.hstack([x_ref_hist, u_ref_hist])

    # Compute optimal control policies, backwards in time
    if not use_robust_lqr:
        # Truncate the G_hist[k] to match the expected format of lqr() i.e. with no adversary blocks
        G_hist_for_lqr = np.zeros([T, n+m+n+m, n+m+n+m])
        for t in range(T):
            Znm, Zmn = np.zeros([n, m]), np.zeros([m, n])
            G_hist_for_lqr[t] = np.block([[Qref+Qorg, Znm, Qorg, Znm],
                                  [Zmn, Rref+Rorg, Zmn, Rorg],
                                  [Qorg, Znm, Qorg, Znm],
                                  [Zmn, Rorg, Zmn, Rorg]])
        K_hist, L_hist, e_hist, P_hist, q_hist, r_hist = lqr(z_hist, A_hist, B_hist, G_hist_for_lqr)
    else:
        lqrm_args = {'z_hist': z_hist,
                     'A_hist': A_hist,
                     'B_hist': B_hist,
                     'C_hist': C_hist,
                     'Ai_hist': Ai_hist,
                     'Bi_hist': Bi_hist,
                     'Ci_hist': Ci_hist,
                     'alpha_var_hist': alpha_var_hist,
                     'beta_var_hist': beta_var_hist,
                     'gamma_var_hist': gamma_var_hist,
                     'G_hist': G_hist,
                     'E_hist': E_hist,
                     'W_hist': W_hist}
        lqrm_outs = lqrm(**lqrm_args)
        K_hist, L_hist, e_hist, Kv_hist, Lv_hist, ev_hist, P_hist, q_hist, r_hist = lqrm_outs

    return LQRController(K_hist, L_hist, e_hist, z_hist, x_ref_hist, u_ref_hist)


if __name__ == "__main__":
    # Open-loop control sequence

    # input_file = "OptTraj_v1_0_1607307033_inputs"
    input_file = 'OptTraj_short_v2_0_1627921766_inputs'

    x_ref_hist, u_ref_hist = load_ref_traj(input_file)

    # Start in the reference initial state
    x0 = x_ref_hist[0]

    # Number of states, inputs
    T, n = x_ref_hist.shape
    T, m = u_ref_hist.shape

    t_hist = np.arange(T) * DT

    # Create controller objects
    ol_controller = OpenLoopController(u_ref_hist)
    lqr_controller = create_lqrm_controller(x_ref_hist, u_ref_hist, use_robust_lqr=False)
    lqrm_controller = create_lqrm_controller(x_ref_hist, u_ref_hist, use_robust_lqr=True)
