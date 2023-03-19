#!/usr/bin/env python3
"""
Changelog:
New in version 1_0:
- Create script

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
Github:
@The-SS

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script does the following:
- Import the `NodeListData` pickle file that RRT* generates
- Extract the optimal trajectory and print its cost
- Shorten the trajectory so that each segment between sampled points is given a proportional horizon to its length (both
 in linear and angular distance)

"""

import os
import math
import copy
import pickle

import numpy as np
import numpy.linalg as la
from casadi import SX, mtimes, vertcat, reshape, nlpsol

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rans_rrtstar.common.dynamics import DYN
from rans_rrtstar.common.tree import TreeNode  # Needed for pickle import of TreeNode objects
from rans_rrtstar.common.drrrts_nmpc import find_dr_padding, get_padded_edges
from rans_rrtstar.common.collision_check import point_obstacle_collision_flag, line_obstacle_collision_flag
from rans_rrtstar import file_version, config

from utility.pickle_io import pickle_import
from rans_rrtstar.config import DT, VELMIN, VELMAX, ANGVELMIN, ANGVELMAX


# Global variables
FILEVERSION = file_version.FILEVERSION  # version of this file

DT = config.DT  # timestep between controls
SAVEPATH = config.SAVEPATH  # path where RRT* data is located and where this data will be stored
GOALAREA = config.GOALAREA  # goal area [xmin,xmax,ymin,ymax]
OBSTACLELIST = config.OBSTACLELIST
SIGMAW = config.SIGMAW
SIGMAV = config.SIGMAV
CROSSCOR = config.CROSSCOR
# QLL = config.QLL
# RLL = config.RLL
QHL = config.QHL
RHL = config.RHL
QTLL = config.QTLL
ROBRAD = config.ROBRAD  # radius of robot (added as padding to environment bounds and the obstacles
OBSTACLELIST = copy.copy(config.OBSTACLELIST)  # [ox,oy,wd,ht]
RANDAREA = copy.copy(config.RANDAREA)  # [xmin,xmax,ymin,ymax]
ALFA = config.ALFA
DRRRT = config.DRRRT

lastalfa = ALFA[-1]
obsalfa = ALFA[0:-4]
obsalfa.insert(0, lastalfa)
ALFA = obsalfa


def get_full_opt_traj_and_ctrls(pathNodesList):
    """
    Extract the full state and control sequence from pathNodesList
    """

    tree_node_inputs = []  # full optimal trajectory inputs
    tree_node_states = []  # full optimal trajectory states
    opt_traj_nodes = []  # only has the optimal trajectory using the sampled points
    found_node_in_goal = False

    if pathNodesList is not None:
        print("Sampled paths exist")

        least_cost_node = []
        found_first_node = False
        # find a node in the goal region
        for node in pathNodesList:
            x = node.means[-1, 0, :][0]
            y = node.means[-1, 1, :][0]
            xmin_goal = GOALAREA[0]
            xmax_goal = GOALAREA[1]
            ymin_goal = GOALAREA[2]
            ymax_goal = GOALAREA[3]
            if (x > xmin_goal) and (x < xmax_goal) and (y > ymin_goal) and (y < ymax_goal):
                found_node_in_goal = True
                if not found_first_node:
                    least_cost_node = node
                    found_first_node = True
                elif node.cost < least_cost_node.cost:
                    least_cost_node = copy.copy(node)
        goal_node = least_cost_node
        # if a node in the goal region is not found, return
        if not found_node_in_goal:
            print("No node in goal region found")
            return
        else:
            print('Found path with cost: ', goal_node.cost)

        # if a node in the goal region is found, construct the optimal trajectory
        traj = []
        ctrl_inputs = []
        num_traj_states = len(goal_node.means)
        node_pt = [goal_node.means[-1, 0, :][0], goal_node.means[-1, 1, :][0], goal_node.means[-1, 2, :][0]]
        for i in range(num_traj_states-1):
            pt = [goal_node.means[i, 0, :][0], goal_node.means[i, 1, :][0], goal_node.means[i, 2, :][0]]
            traj.append(pt)
            ctrl = [goal_node.input_commands[i, 0], goal_node.input_commands[i, 1]]
            ctrl_inputs.append(ctrl)
            pt_2 = [goal_node.means[i+1, 0, :][0], goal_node.means[i+1, 1, :][0], goal_node.means[i+1, 2, :][0]]
            diff = DYN.one_step_sim_diff(pt, ctrl, pt_2)
            if np.sum(diff) > 1e-14:
                print(diff)
        opt_traj_nodes = [node_pt] + opt_traj_nodes
        tree_node_states = traj + tree_node_states
        tree_node_inputs = ctrl_inputs + tree_node_inputs
        # find index of parent
        idx_of_parent_node = goal_node.parent
        prev_parent_state = [goal_node.means[0, 0, :][0], goal_node.means[0, 1, :][0], goal_node.means[0, 2, :][0]]
        while idx_of_parent_node != None: # if parent found
            parent_node = pathNodesList[idx_of_parent_node] # get parent node
            # add parent node info to data
            traj = []
            ctrl_inputs = []
            node_pt = [parent_node.means[-1, 0, :][0], parent_node.means[-1, 1, :][0], parent_node.means[-1, 2, :][0]]
            num_traj_states = len(parent_node.means)

            current_parent_state = node_pt
            diff = np.abs(np.array(current_parent_state) - np.array(prev_parent_state))
            if np.sum(diff) > 1e-14:
                print(diff)
                print('~~~~~')
            prev_parent_state = [parent_node.means[0, 0, :][0], parent_node.means[0, 1, :][0], parent_node.means[0, 2, :][0]]
            for i in range(num_traj_states-1):
                pt = [parent_node.means[i, 0, :][0], parent_node.means[i, 1, :][0], parent_node.means[i, 2, :][0]]
                traj.append(pt)
                ctrl = [parent_node.input_commands[i, 0], parent_node.input_commands[i, 1]]
                ctrl_inputs.append(ctrl)
                pt_2 = [parent_node.means[i + 1, 0, :][0], parent_node.means[i + 1, 1, :][0],
                        parent_node.means[i + 1, 2, :][0]]
                diff = DYN.one_step_sim_diff(pt, ctrl, pt_2)
                if np.sum(diff) > 1e-14:
                    print(diff)
            opt_traj_nodes = [node_pt] + opt_traj_nodes
            tree_node_states = traj + tree_node_states
            tree_node_inputs = ctrl_inputs + tree_node_inputs
            # find index of parent
            idx_of_parent_node = parent_node.parent
    print('Number of steps: ', len(np.array(tree_node_states)))

    DYN.check_entire_traj(tree_node_states, tree_node_inputs)

    return [np.array(opt_traj_nodes), np.array(tree_node_states), np.array(tree_node_inputs)]


def get_sampled_traj_and_ctrls(pathNodesList):
    """
    Extract the nodes and control sequence at each node from pathNodesList
    """
    start_state = []
    control_inputs = []
    state_trajectory = []
    for k, node in enumerate(pathNodesList):
        point = [node.means[-1, 0, :][0], node.means[-1, 1, :][0], node.means[-1, 2, :][0]]
        ctrls = node.input_commands
        if k == 0:
            start_state.append(point)
            state_trajectory.append(point)
            control_inputs.append(ctrls)
        else:
            state_trajectory.append(point)
            control_inputs.append(ctrls)

    tree_states, tree_ctrl = reshape_data(state_trajectory, control_inputs, 3)  # 3 = num states

    return [start_state, tree_states, tree_ctrl]


def reshape_data(state_trajectory, control_inputs, numstates):
    """
    Reshapes the data of get_sampled_traj_and_ctrls
    """
    traj = np.array(state_trajectory)
    len_traj = len(state_trajectory)
    traj = traj.reshape(len_traj, numstates)
    ctrl = np.array(control_inputs)
    return [traj, ctrl]


def plot_data(tree_states, sampled_opt_traj_nodes_, full_opt_traj_states, full_opt_traj_ctrls,
              new_filename, save_opt_path_plot, title_str=''):
    """
    plots a figure (and saves it) with the extracted optimal trajectory and inputs along the heading direction
    """

    # all sampled points
    x_sampled = tree_states[:, 0]
    y_sampled = tree_states[:, 1]

    # select nodes for optimal trajectory from sampled points
    x_sampled_opt = sampled_opt_traj_nodes_[:, 0]
    y_sampled_opt = sampled_opt_traj_nodes_[:, 1]

    # all optimal x,y trajectory pairs
    x = full_opt_traj_states[:, 0]
    y = full_opt_traj_states[:, 1]

    # plot goal
    x1mingoal = GOALAREA[0]
    x1maxgoal = GOALAREA[1]
    y1mingoal = GOALAREA[2]
    y1maxgoal = GOALAREA[3]
    goalHeight = y1mingoal
    xGoal = np.arange(x1mingoal, x1maxgoal, 0.2).tolist()  # [-5.0,-4.8, -4.6, -4.4, -4.2, -4.0]
    y1Goal = [goalHeight]

    # Shade the area between y1 and line y=y1maxgoal
    fig, ax = plt.subplots()
    plt.fill_between(xGoal, y1Goal, y1maxgoal,
                     facecolor="#CAB8CB",  # The fill color
                     color='#CAB8CB')  # The outline color

    # Plot the obstacles
    obstacles = [Rectangle(xy=[ox, oy],
                           width=wd ,
                           height=ht,
                           angle=0,
                           color="k") for (ox, oy, wd, ht) in OBSTACLELIST]
    for obstacle in obstacles:
        ax.add_artist(obstacle)

    ax.add_artist(Rectangle(xy=[-5, -5], width=0.1, height=10, angle=0, color="k"))
    ax.add_artist(Rectangle(xy=[-5, -5], width=10, height=0.1, angle=0, color="k"))
    ax.add_artist(Rectangle(xy=[-5, 4.9], width=10, height=0.1, angle=0, color="k"))
    ax.add_artist(Rectangle(xy=[4.9, -5], width=0.1, height=10, angle=0, color="k"))
    # xy, w, h = (-5.0, -5.0), 10.0, 10.0  # (-1.2, -0.2), 2.4, 2.2
    # r = Rectangle(xy, w, h, fc='none', ec='gold', lw=1)
    # ax.add_artist(r)

    # plot sampled points, selected samples for optimal trajectory, and optimal trajectory

    ax.plot(x_sampled, y_sampled, 'o', color='indianred')
    ax.plot(x_sampled_opt, y_sampled_opt, 'x', color='black')
    ax.plot(x, y)

    # plot vehicle heading and velocity input
    for i, [v, w] in enumerate(full_opt_traj_ctrls):
        # vehicle state
        x_veh = full_opt_traj_states[i, 0]
        y_veh = full_opt_traj_states[i, 1]
        theta_veh = full_opt_traj_states[i,2]
        dx = v*math.cos(theta_veh)
        dy = v*math.sin(theta_veh)
        ax.arrow(x_veh, y_veh, dx, dy, head_width=0.05, fc='k', ec='c')

    ax.set_title(title_str)

    plot_name = new_filename + '_plot.png'
    if save_opt_path_plot:
        plt.savefig(plot_name)

    plt.show()


def save_data(full_opt_traj_states, full_opt_traj_ctrls, new_filename):
    # TODO replace with pickle_export, need directory name
    state_file = new_filename + "_states"
    outfile = open(state_file, 'wb')
    pickle.dump(full_opt_traj_states, outfile)
    outfile.close()
    inputs_file = new_filename + "_inputs"
    outfile = open(inputs_file, 'wb')
    pickle.dump(full_opt_traj_ctrls, outfile)
    outfile.close()


###############################################################################
# TODO: automate making setup_steering_law_parameters and nonlinsteer match with rrstar.py functions

# def SetUpSteeringLawParametersSHORTENING(N, T, v_max, v_min, omega_max, omega_min, x_max=np.inf, x_min=-np.inf, y_max=np.inf, y_min=-np.inf, theta_max=np.inf, theta_min=-np.inf):
#     Q = QHL
#     R = RHL
#
#     opti = casadi.Opti()
#
#     # Define symbolic states using Casadi Opti
#     x = opti.variable()
#     y = opti.variable()
#     theta = opti.variable()
#     states = vertcat(x, y, theta)  # all three states
#     n_states = states.size()[0]  # number of symbolic states
#
#     # Define symbolic inputs using Cadadi SX
#     v = opti.variable()
#     omega = opti.variable()
#     controls = vertcat(v, omega)  # both controls
#     n_controls = controls.size()[0]  # number of symbolic inputs
#
#     # RHS of nonlinear unicycle dynamics (continuous time model)
#     rhs = horzcat(v * cos(theta), v * sin(theta), omega)
#
#     # Unicycle continuous time dynamics function
#     f = Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])
#
#     # Casadi Opti trajectory variables/parameters for multiple shooting
#     U = opti.variable(N, n_controls)  # N trajectory controls
#     X = opti.variable(N + 1, n_states)  # N+1 trajectory states
#     P = opti.parameter(2, n_states)  # first and last states as independent parameters
#
#     # Cost function
#     obj = 0  # objective/cost
#     opti.subject_to(X[0, :].T == P[0, :].T)  # match first (current) state
#     opti.subject_to(X[N, :].T == P[1, :].T)  # match last (desired) state
#     for i in range(N):
#         # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
#         obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
#
#         # compute the next state from the dynamics
#         x_next_ = f(X[i, :], U[i, :]) * T + X[i, :]
#
#         # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting) (satisfy dynamics)
#         opti.subject_to(X[i + 1, :].T == x_next_.T)
#
#     opti.minimize(obj)
#     opti.subject_to(opti.bounded(x_min, X[:, 0], x_max))
#     opti.subject_to(opti.bounded(y_min, X[:, 1], y_max))
#     opti.subject_to(opti.bounded(theta_min, X[:, 2], theta_max))
#     # input constraints
#     opti.subject_to(opti.bounded(v_min, U[:, 0], v_max))
#     opti.subject_to(opti.bounded(omega_min, U[:, 1], omega_max))
#
#     opts_setting = {'ipopt.max_iter': 2000,
#                     'ipopt.print_level': 0,  # 4
#                     'print_time': 0,
#                     'verbose': 0,  # 1
#                     'error_on_fail': 1}
#
#     opti.solver("ipopt", opts_setting)
#
#     solver = opti
#
#     DELTA = []
#     OBSPAD, ENVPAD = [], []
#     return solver, f, n_states, n_controls, U, X, P, DELTA, OBSPAD, ENVPAD
#
# def nonlinsteerSHORTENING(solver, x0, xT, n_states, n_controls, N,
#                           T, U, X, P, DELTA, OBSPAD, ENVPAD, current_ref_traj,
#                           current_ref_inputs, obs_pad, env_pad):
#
#     solver.set_initial(X, current_ref_traj)
#     solver.set_initial(U, current_ref_inputs)
#     x0 = list(x0.reshape(n_states))
#     xT = list(xT.reshape(n_states))
#     x0xT = np.array([x0, xT])
#     solver.set_value(P, x0xT)
#
#     try:
#         res = solver.solve()
#     except:
#         print('Steering NLP Failed')
#         return [], []
#
#     u_casadi = res.value(U) # shape: (N, n_controls)
#     # Get the predicted state trajectory for N time steps ahead
#     x_casadi = res.value(X) # shape: # (N+1, n_states)
#
#     return x_casadi, u_casadi

def SetUpSteeringLawParameters(N, T, v_max, v_min, omega_max, omega_min, x_max=np.inf, x_min=-np.inf, y_max=np.inf, y_min=-np.inf, theta_max=np.inf, theta_min=-np.inf):
    """
    Sets up an IPOPT NLP solver using Casadi SX
    Inputs:
        N: horizon
        T: time step (sec)
        v_max, v_min: maximum and minimum linear velocities in m/s
        omega_max, omega_min: maximum and minimum angular velocities in rad/s
        x_max, x_min, y_max, y_min, theta_max, theta_min: max and min bounds on the states x, y, and theta
    Outputs:
        solver: Casadi NLP solver using ipopt
        f: Casadi continuous time dynamics function
        n_states, n_controls: number of states and controls
        lbx, ubx, lbg, ubg: lower and upper (l,u) state and input (x,g) bounds
    """

    # Define state and input cost matrices
    Q = QHL
    # np.array([[1.0, 0.0, 0.0],
    #               [0.0, 1.0, 0.0],
    #               [0.0, 0.0, 0.001]])
    R = RHL
    # 100 * np.array([[1.0, 0.0],
    #                     [0.0, 1.0]])



    # Unicycle continuous time dynamics function
    f = DYN.ctime_dynamics_cas()

    # Casadi SX trajectory variables/parameters for multiple shooting
    U = SX.sym('U', N, DYN.m)  # N trajectory controls
    X = SX.sym('X', N + 1, DYN.n)  # N+1 trajectory states
    P = SX.sym('P', DYN.n + DYN.n)  # first and last states as independent parameters

    # Concatinate the decision variables (inputs and states)
    opt_variables = vertcat(reshape(U, -1, 1), reshape(X, -1, 1))

    # Cost function
    obj = 0  # objective/cost
    g = []  # equality constraints
    g.append(X[0, :].T - P[:3])  # add constraint on initial state
    for i in range(N):
        # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
        # obj += 0*mtimes([(X[i, :]-P[3:].T), Q, (X[i, :]-P[3:].T).T])  # quadratic penalty on state deviation from target
        obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort

        # compute the next state from the dynamics
        x_next_ = f(X[i, :], U[i, :]) * T + X[i, :]

        # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting)
        g.append(X[i + 1, :].T - x_next_.T)

    g.append(X[N, :].T - P[3:])  # constraint on final state
    # g.append(X[N, 0:2].T - P[3:5])  # constraint on final state
    # obj = obj+mtimes([(X[N, :]-P[3:].T), Q, (X[N, :]-P[3:].T).T])  # final state cost

    # Set the nlp problem
    nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': vertcat(*g)}

    opts_setting = {'ipopt.max_iter': 2000,
                    'ipopt.print_level': 0, #4
                    'print_time': 0,
                    'verbose': 0, # 1
                    'error_on_fail': 1}

    # Create a solver that uses IPOPT with above solver settings
    solver = nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    # Define the bounds on states and controls
    lbx = []
    ubx = []
    lbg = 0.0
    ubg = 0.0
    # Upper and lower bounds on controls
    for _ in range(N):
        lbx.append(v_min)
        ubx.append(v_max)
    for _ in range(N):
        lbx.append(omega_min)
        ubx.append(omega_max)
    # Upper and lower bounds on states
    for _ in range(N + 1):
        lbx.append(x_min)
        ubx.append(x_max)
    for _ in range(N + 1):
        lbx.append(y_min)
        ubx.append(y_max)
    for _ in range(N + 1):
        lbx.append(theta_min)
        ubx.append(theta_max)

    return solver, f, DYN.n, DYN.m, lbx, ubx, lbg, ubg


def nonlinsteer(solver, x0, xT, n_states, n_controls, N, T, lbg, lbx, ubg, ubx):
    """
    Solves the nonlinear steering problem using the solver from setup_steering_law_parameters
    Inputs:
        solver: Casadi NLP solver from setup_steering_law_parameters
        x0, xT: initial and final states as (n_states)x1 ndarrays e.g. [[2.], [4.], [3.14]]
        n_states, n_controls: number of states and controls
        N: horizon
        T: time step
        lbg, lbx, ubg, ubx:  lower and upper (l,u) state and input (x,g) bounds
    Outputs:
        x_casadi, u_casadi: trajectory states and inputs returned by Casadi
            if solution found:
                states: (N+1)x(n_states) ndarray e.g. [[1  2  0], [1.2  2.4  0], [2  3.5  0]]
                controls: (N)x(n_controls) ndarray e.g. [[0.5  0], [1  0.01], [1.2  -0.01]]
            else, [],[] returned
    """

    # Create an initial state trajectory that roughly accomplishes the desired state transfer (by interpolating)
    init_states_param = np.linspace(0, 1, N + 1)
    init_states = np.zeros([N + 1, n_states])
    dx = xT - x0
    for i in range(N + 1):
        init_states[i] = (x0 + init_states_param[i] * dx).flatten()

    # Create an initial input trajectory that roughly accomplishes the desired state transfer
    # (using interpolated states to compute rough estimate of controls)
    dist = la.norm(xT[0:2] - x0[0:2])
    ang_dist = xT[2][0] - x0[2][0]
    total_time = N * T
    const_vel = dist / total_time
    const_ang_vel = ang_dist / total_time
    init_inputs = np.array([const_vel, const_ang_vel] * N).reshape(-1, 2)

    ## set parameter
    c_p = np.concatenate((x0, xT))  # start and goal state constraints
    init_decision_vars = np.concatenate((init_inputs.reshape(-1, 1), init_states.reshape(-1, 1)))
    try:
        res = solver(x0=init_decision_vars, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    except:
        # raise Exception('NLP failed')
        # print('NLP Failed')
        return [], []

    # Output series looks like [u0, u1, ..., uN, x0, x1, ..., xN+1]  #####[u0, x0, u1, x1, ...]
    casadi_result = res['x'].full()

    # Extract the control inputs and states
    u_casadi = casadi_result[:2 * N].reshape(n_controls, N).T  # (N, n_controls)
    x_casadi = casadi_result[2 * N:].reshape(n_states, N + 1).T  # (N+1, n_states)

    return x_casadi, u_casadi


def nonlinsteerNoColAvoid(solver, x0, xT, n_states, n_controls, N, T, U, X, P, DELTA, OBSPAD, ENVPAD, current_ref_traj, current_ref_inputs, obs_pad, env_pad):
    """
    Solves the nonlinear steering problem using the solver from SetUpSteeringLawParametersBigM
    Inputs:
        solver: Casadi NLP solver from setup_steering_law_parameters
        x0, xT: initial and final states as (n_states)x1 ndarrays e.g. [[2.], [4.], [3.14]]
        n_states, n_controls: number of states and controls
        N: horizon
        T: time step
        lbg, lbx, ubg, ubx:  lower and upper (l,u) state and input (x,g) bounds
        current_ref_traj, current_ref_inputs: reference trajectory and reference inputs as Nx(n_states) ndarrays# TODO: add shapes
    Outputs:
        x_casadi, u_casadi: trajectory states and inputs returned by Casadi
            if solution found:
                states: (N+1)x(n_states) ndarray e.g. [[1  2  0], [1.2  2.4  0], [2  3.5  0]]
                controls: (N)x(n_controls) ndarray e.g. [[0.5  0], [1  0.01], [1.2  -0.01]]
            else, [],[] returned
    """

    # Create an initial state trajectory that roughly accomplishes the desired state transfer (by interpolating)
    init_states_param = np.linspace(0, 1, N + 1)
    init_states = np.zeros([N + 1, n_states])
    dx = xT - x0
    for i in range(N + 1):
        init_states[i] = (x0 + init_states_param[i] * dx).flatten()

    # Create an initial input trajectory that roughly accomplishes the desired state transfer
    # (using interpolated states to compute rough estimate of controls)
    dist = la.norm(xT[0:2] - x0[0:2])
    ang_dist = xT[2][0] - x0[2][0]
    total_time = N * T
    const_vel = dist / total_time
    const_ang_vel = ang_dist / total_time
    init_inputs = np.array([const_vel, const_ang_vel] * N).reshape(-1, 2)

    ## set parameter
    constraint_states = []
    constraint_states.append(x0.reshape(n_states))

    for ref_state in current_ref_traj:
        constraint_states.append(ref_state.reshape(n_states))

    init_inputs = []
    for ref_input in current_ref_inputs:
        init_inputs.append(ref_input.reshape(n_controls))
    init_inputs = np.array(init_inputs)

    constraint_states = np.array(constraint_states)
    solver.set_value(P, constraint_states)
    solver.set_initial(X, constraint_states)
    solver.set_initial(U, init_inputs)
    # solver.set_initial(X, init_states)
    # solver.set_initial(U, init_inputs)
    try:
        res = solver.solve()
    except:
        print('Steering NLP Failed')
        return [], []

    # Update the cost_total
    # cost_total = res.value(self.obj)  # self.opti.debug.value(self.obj)
    # Obtain the optimal control input sequence
    u_casadi = res.value(U) # shape: (N, n_controls)
    # Get the predicted state trajectory for N time steps ahead
    x_casadi = res.value(X) # shape: # (N+1, n_states)

    return x_casadi, u_casadi


def GetDynamics(N):
    # Nonlinear State Update function f(x,u)
    # Given {states, controls} as inputs, returns {rhs} as output
    f = DYN.ctime_dynamics_cas()

    # Input and state constraints
    v_max = VELMAX  # maximum linear velocity (m/s)
    v_min = VELMIN  # minimum linear velocity (m/s)
    omega_max = ANGVELMAX  # 0.125 * (2 * np.pi)  # maximum angular velocity (rad/s)
    omega_min = ANGVELMIN  # minimum angular velocity (rad/s)
    x_max = RANDAREA[1]  # maximum state in the horizontal direction
    x_min = RANDAREA[0]  # minimum state in the horizontal direction
    y_max = RANDAREA[3]  # maximum state in the vertical direction
    y_min = RANDAREA[2]  # minimum state in the vertical direction
    theta_max = np.inf  # maximum state in the theta direction
    theta_min = -np.inf  # minimum state in the theta direction

    lbx = []
    ubx = []
    lbg = 0.0
    ubg = 0.0
    # Upper and lower bounds on controls
    for _ in range(N):
        lbx.append(v_min)
        ubx.append(v_max)
    for _ in range(N):
        lbx.append(omega_min)
        ubx.append(omega_max)
    # Upper and lower bounds on states
    for _ in range(N + 1):
        lbx.append(x_min)
        ubx.append(x_max)
    for _ in range(N + 1):
        lbx.append(y_min)
        ubx.append(y_max)
    for _ in range(N + 1):
        lbx.append(theta_min)
        ubx.append(theta_max)

    # Create the arguments dictionary to hold the constraint values
    argums = {'lbg': lbg,
              'ubg': ubg,
              'lbx': lbx,
              'ubx': ubx}

    return argums, f


def SetUpSteeringLawParametersWithFinalHeading(N, dt, numStates, numControls):
    """
    Same function as that in tree.py
    """

    # Get the dynamics specific data
    argums, f = GetDynamics(N)

    # Define state and input cost matrices for solving the NLP
    Q = QHL
    R = RHL

    # Casadi SX trajectory variables/parameters for multiple shooting
    U = SX.sym('U', N, numControls)  # N trajectory controls
    X = SX.sym('X', N + 1, numStates)  # N+1 trajectory states
    P = SX.sym('P', numStates + numStates)  # first and last states as independent parameters

    # Concatinate the decision variables (inputs and states)
    opt_variables = vertcat(reshape(U, -1, 1), reshape(X, -1, 1))

    # Cost function
    obj = 0  # objective/cost
    g = []  # equality constraints
    g.append(X[0, :].T - P[:3])  # add constraint on initial state
    for i in range(N):
        # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
        obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
        # compute the next state from the dynamics
        x_next_ = f(X[i, :], U[i, :]) * dt + X[i, :]
        # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting)
        g.append(X[i + 1, :].T - x_next_.T)
    g.append(X[N, 0:3].T - P[3:])  # constraint on final state including the heading angle

    # Set the nlp problem
    nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': vertcat(*g)}

    # Set the nlp problem settings
    opts_setting = {'ipopt.max_iter': 2000,
                    'ipopt.print_level': 0,  # 4
                    'print_time': 0,
                    'verbose': 0,  # 1
                    'error_on_fail': 1}

    # Create a solver that uses IPOPT with above solver settings
    solver = nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    # self.SteerSetParams2 = SteerSetParams2(dt, f, N, solver, argums, numStates, numControls)
    return solver, argums


def nonlinsteerRANSRRTSTAR(x0, xGoal, N, solver, argums, numStates, numControls):

    [x_casadi, u_casadi] = solveNLP(solver, argums, x0, xGoal, numStates, numControls, N, DT)
    if x_casadi == []:  # NLP problem failed to find a solution
        return False, [], []
    else:
        return True, x_casadi, u_casadi


def solveNLP(solver, argums, x0, xT, n_states, n_controls, N, T):
    # Create an initial state trajectory that roughly accomplishes the desired state transfer (by interpolating)
    init_states_param = np.linspace(0, 1, N + 1)
    init_states = np.zeros([N + 1, n_states])
    dx = xT - x0
    for i in range(N + 1):
        init_states[i] = (x0 + init_states_param[i] * dx).flatten()

    # Create an initial input trajectory that roughly accomplishes the desired state transfer
    # (using interpolated states to compute rough estimate of controls)
    dist = la.norm(xT[0:2] - x0[0:2])
    ang_dist = xT[2] - x0[2]
    total_time = N * T
    const_vel = dist / total_time
    const_ang_vel = ang_dist / total_time
    init_inputs = np.array([const_vel, const_ang_vel] * N).reshape(-1, 2)

    # Initialize and solve NLP
    c_p = np.concatenate((x0, xT))  # start and goal state constraints
    init_decision_vars = np.concatenate((init_inputs.reshape(-1, 1), init_states.reshape(-1, 1)))
    try:
        res = solver(x0=init_decision_vars, p=c_p, lbg=argums['lbg'], lbx=argums['lbx'], ubg=argums['ubg'],
                     ubx=argums['ubx'])
    except:
        # raise Exception('NLP failed')
        # print('NLP Failed')
        return [], []

    # Output series looks like [u0, u1, ..., uN, x0, x1, ..., xN+1]  #####[u0, x0, u1, x1, ...]
    casadi_result = res['x'].full()

    # Extract the control inputs and states
    u_casadi = casadi_result[:2 * N].reshape(n_controls, N).T  # (N, n_controls)
    x_casadi = casadi_result[2 * N:].reshape(n_states, N + 1).T  # (N+1, n_states)

    return x_casadi, u_casadi

def create_and_add_solver(N, dt, n, m, solvers_list, solvers_horizons, solvers_params):
    solver, argums = SetUpSteeringLawParametersWithFinalHeading(N, dt, n, m)
    solvers_list.append(solver)
    solvers_horizons.append(N)
    solvers_params.append(argums)
    return solver, argums, solvers_list, solvers_horizons, solvers_params


def get_or_create_solver(N, dt, n, m, solvers_list, solvers_horizons, solvers_params):
    # check if solver with N states exists
    if N in solvers_horizons:  # solver already exists
        solver_index = solvers_horizons.index(N)
        solver = solvers_list[solver_index]
        params = solvers_params[solver_index]
    else:
        solver, params, solvers_list, solvers_horizons, solvers_params = create_and_add_solver(N, dt, n, m, solvers_list, solvers_horizons, solvers_params)
    return solver, params, solvers_list, solvers_horizons, solvers_params


def shorten_traj(sampled_opt_traj_nodes_, all_rrt_states, all_rrt_inputs):
    T, omega_max, v_max, num_states, num_controls = DT, ANGVELMAX, VELMAX, DYN.n, DYN.m

    short_states = []
    short_ctrls = []

    solvers_list = []
    solvers_horizons = []
    solvers_params = []

    for node_idx, node_state in enumerate(sampled_opt_traj_nodes_):
        # first state --> just start there
        if node_idx == 0:
            short_states.append(node_state)
            continue
        # otherwise, go from last node to this node
        from_state = short_states[-1]
        to_state = node_state

        # estimate horizon needed
        linear_distance = la.norm(from_state[0:2] - to_state[0:2])
        angular_distance = abs(to_state[2] - from_state[2])
        # horizon for translation = (translation amount (units)) / (translation speed (units/sec) * step time (sec))
        N_translation = linear_distance / (v_max * T)
        # horizon for rotation = (rotation amount (rad)) / (rotation speed (rad/sec) * step time (sec))
        N_rotation = angular_distance / (omega_max * T)
        # upper bound on total horizon
        N = int((N_rotation + N_translation) * 1.5)

        solver, argums, solvers_list, solvers_horizons, solvers_params = get_or_create_solver(N, T, num_states, num_controls, solvers_list, solvers_horizons, solvers_params)

        # try to steer
        steer_success = False
        while not steer_success:
            steer_success, x_casadi, u_casadi = nonlinsteerRANSRRTSTAR(from_state, to_state, N, solver, argums, num_states, num_controls)
            if steer_success:
                if DRRRT:
                    # check for DR collisions
                    prev_pt = x_casadi[0]
                    obstaclelist, envbounds = OBSTACLELIST, RANDAREA
                    robrad = ROBRAD
                    safe_check = True
                    obs_edges, _ = get_padded_edges()
                    env_pad, obs_pad = find_dr_padding(ALFA, N, obs_edges, [SIGMAW] * (N + 1))
                    for pt_idx, point in enumerate(x_casadi):
                        # get dr-padding value
                        destination_dr_padding = obs_pad[-1, -1]  # this assumes all obstacles have equal padding
                        # check if new node is safe
                        col_flag = point_obstacle_collision_flag(point, obstaclelist, envbounds, robrad + destination_dr_padding)
                        if col_flag:
                            safe_check = False
                            break
                        # check if line connecting it to previous node is safe
                        col_flag = line_obstacle_collision_flag(prev_pt, point, obstaclelist, robrad + destination_dr_padding)
                        prev_pt = point
                        if col_flag:
                            safe_check = False
                            break
                    if not safe_check:
                        steer_success = False
                    else:
                        break
            else:
                N += 1
                solver, argums, solvers_list, solvers_horizons, solvers_params = get_or_create_solver(N, T, num_states,
                                                                                                      num_controls,
                                                                                                      solvers_list,
                                                                                                      solvers_horizons,
                                                                                                      solvers_params)

            # if N == STEER_TIME: # USE ALREADY COMPUTED VALUES
        # save the new states and controls
        for j, ctrl in enumerate(u_casadi):
            short_states.append(x_casadi[j + 1])
            short_ctrls.append(ctrl)

        failures = DYN.check_entire_traj(x_casadi, u_casadi, 1e-7)
        if failures[0]:
            print("Failed Simulation Check:", failures)

    failures = DYN.check_entire_traj(short_states, short_ctrls, 1e-7)
    print("Failed Simulation Check:", failures)

    num_traj_states = len(short_states)
    new_states = np.array(short_states).reshape(num_traj_states, num_states)
    new_controls = np.array(short_ctrls).reshape(num_traj_states - 1, num_controls)
    print('Number of steps: ', num_traj_states)
    return new_states, new_controls


# def shorten_trajOLD(sampled_opt_traj_nodes_, all_rrt_states, all_rrt_inputs, T, omega_max, v_max, num_states, num_controls):
#     from sim_check import check_entire_traj, one_step_sim
#     dr_short = DRRRT  # True --> DR version, False --> no DR
#     new_states = []
#     new_controls = []
#
#     created_solvers = []
#     horizon_of_solvers = []
#     all_solver_params = []
#     for node_idx, state in enumerate(sampled_opt_traj_nodes_):
#         if node_idx == 0: # first node
#             new_states.append(state)
#             continue
#         from_state = new_states[-1] # get previous state
#         to_state = state # get next state
#
#         # estimate the horizon required to go from from_state to to_state
#         linear_distance = la.norm(from_state[0:2] - to_state[0:2])
#         angular_distance = abs(to_state[2] - from_state[2])
#         N_translation = linear_distance / (v_max * T) # horizon required for translation = (translation amount (units)) / (translation speed (units/sec) * step time (sec))
#         N_rotation = angular_distance / (omega_max * T) # horizon required for rotation = (rotation amount (rad)) / (rotation speed (rad/sec) * step time (sec))
#         N = int(ceil(N_rotation + N_translation)*2)
#
#         solved = False
#         while not solved:
#             # check if a solver with N has been created before
#             try:
#                 solver_index = horizon_of_solvers.index(N)
#                 # if found, get the solver
#                 solver = created_solvers[solver_index]
#                 solver_params = all_solver_params[solver_index]
#             except:
#                 if not dr_short:  # RRT* is originally not DR-RRT*
#                     # if not found, create the solver
#                     [solver, _, _, _, lbx, ubx, lbg, ubg] = setup_steering_law_parameters(N, T, v_max, -v_max,
#                                                                                            omega_max, -omega_max,
#                                                                                            5, -5,
#                                                                                            5, -5,
#                                                                                            np.inf, -np.inf)
#                 else:  # RRT* is a DR-RRT*
#                     # [solver, f, _, _, U, X, P, DELTA, OBSPAD, ENVPAD] = SetUpSteeringLawParametersNoColAvoid(N, T, VELMAX,
#                     #                                                                                    VELMIN,
#                     #                                                                                    ANGVELMAX,
#                     #                                                                                    ANGVELMIN)
#                     [solver, _, _, _, U, X, P, DELTA, OBSPAD, ENVPAD] = SetUpSteeringLawParametersSHORTENING(N, T,
#                                                                                                              VELMAX,
#                                                                                                              VELMIN,
#                                                                                                              omega_max,
#                                                                                                              -omega_max,
#                                                                                                              RANDAREA[1],
#                                                                                                              RANDAREA[0],
#                                                                                                              RANDAREA[3],
#                                                                                                              RANDAREA[2],
#                                                                                                              ANGVELMAX,
#                                                                                                              ANGVELMIN)
#                 horizon_of_solvers.append(N) # save horizon length
#                 created_solvers.append(solver) # save solver
#                 if not dr_short:
#                     solver_params = [lbx, ubx, lbg, ubg]
#                 else:
#                     solver_params = [U, X, P, DELTA, OBSPAD, ENVPAD]
#                 all_solver_params.append(solver_params) # save solver params
#
#             # solve the problem
#             if not dr_short:  # RRT* not DR-RRT*
#                 lbx = solver_params[0]
#                 ubx = solver_params[1]
#                 lbg = solver_params[2]
#                 ubg = solver_params[3]
#                 x_casadi, u_casadi = nonlinsteer(solver, from_state.reshape(num_states,1), to_state.reshape(num_states,1),
#                                                  num_states, num_controls, N, T, lbg, lbx, ubg, ubx)
#             else:  # DR-RRT*
#                 U, X, P, DELTA, OBSPAD, ENVPAD = solver_params[0], solver_params[1], solver_params[2], solver_params[3], solver_params[4], solver_params[5]
#
#                 # Create an initial state trajectory that roughly accomplishes the desired state transfer (by interpolating)
#                 x0 = from_state.reshape(num_states,1)
#                 xT = to_state.reshape(num_states,1)
#                 init_states_param = np.linspace(0, 1, N + 1)
#                 init_states = np.zeros([N + 1, num_states])
#                 dx = xT - x0
#                 for i in range(N + 1):
#                     init_states[i] = (x0 + init_states_param[i] * dx).flatten()
#                 # Create an initial input trajectory that roughly accomplishes the desired state transfer
#                 # (using interpolated states to compute rough estimate of controls)
#                 dist = la.norm(xT[0:2] - x0[0:2])
#                 ang_dist = xT[2][0] - x0[2][0]
#                 total_time = N * T
#                 const_vel = dist / total_time
#                 const_ang_vel = ang_dist / total_time
#                 init_inputs = np.array([const_vel, const_ang_vel] * N).reshape(-1, 2)
#                 # get estimated DR-paddings
#                 obs_edges, _ = get_padded_edges()
#                 env_pad, obs_pad = find_dr_padding(ALFA, N, obs_edges, [SIGMAW]*(N+1))
#                 # x_casadi, u_casadi = nonlinsteerNoColAvoid(solver, x0, xT, num_states, num_controls, N,
#                 #                                            T, U, X, P, DELTA, OBSPAD, ENVPAD, init_states[:-1],
#                 #                                            init_inputs, obs_pad, env_pad)
#                 x_casadi, u_casadi = nonlinsteerSHORTENING(solver, x0, xT, num_states, num_controls, N,
#                                                            T, U, X, P, DELTA, OBSPAD, ENVPAD, init_states,
#                                                            init_inputs, obs_pad, env_pad)
#
#                 # now check for DR collisions
#                 if x_casadi != []:
#                     prev_pt = x_casadi[0]
#                     obstaclelist, envbounds = OBSTACLELIST, RANDAREA
#                     robrad = ROBRAD
#                     safe_check = True
#                     for pt_idx, point in enumerate(x_casadi):
#                         # get dr-padding value
#                         destination_dr_padding = obs_pad[-1,-1] # this assumes all obstacles have equal padding
#                         # check if new node is safe
#                         col_flag = point_obstacle_collision_flag(point, obstaclelist, envbounds, robrad+destination_dr_padding)
#                         if col_flag:
#                             safe_check = False
#                             break
#                         # check if line connecting it to previous node is safe
#                         col_flag = line_obstacle_collision_flag(prev_pt, point, obstaclelist, robrad+destination_dr_padding)
#                         if col_flag:
#                             safe_check = False
#                             break
#                 else:
#                     safe_check = False
#
#                 # if not safe_check:
#                 #     x_casadi = [] # consider problem not solved (and hence increment N)
#
#             # if problem not solved, increase N and try again
#             if (x_casadi == [] or not safe_check) and N < STEER_TIME:
#                 N += 1
#                 print(N)
#                 continue
#             else:
#                 solved = True
#
#             if (x_casadi == [] or not safe_check) and N >= STEER_TIME:
#                 print('$$$$$$$$$$$$$$')
#                 print('NLP FAILED')
#                 print(N)
#                 x_casadi = all_rrt_states[(node_idx-1)*STEER_TIME:(node_idx)*STEER_TIME]
#                 u_casadi = all_rrt_inputs[(node_idx-1)*STEER_TIME:(node_idx)*STEER_TIME-1]
#                 solved = True
#                 x_now_actual = np.array(new_states[-1])
#                 x_now_casadi = np.array(x_casadi[0])
#                 x_next_casadi = np.array(x_casadi[1])
#                 u_now_casadi = np.array(u_casadi[0])
#                 print("Difference between current state = ", np.abs(x_now_actual - x_now_casadi))
#                 diff = one_step_sim(x_now_actual, u_now_casadi, x_next_casadi)
#                 print("Difference in sim = ", diff)
#             else:
#                 solved = True
#
#         # save the new states and controls
#         for j, ctrl in enumerate(u_casadi):
#             new_states.append(x_casadi[j+1])
#             new_controls.append(ctrl)
#             # steering_cost += ctrl.dot(R).dot(ctrl)
#
#
#         failures = check_entire_traj(x_casadi, u_casadi, 1e-10)
#         if failures[0]:
#             print(failures)
#
#     # print("Shortened path cost: ", steering_cost)
#
#     failures = check_entire_traj(new_states, new_controls)
#     print(failures)
#
#     num_traj_states = len(new_states)
#     new_states = np.array(new_states).reshape(num_traj_states, num_states)
#     new_controls = np.array(new_controls).reshape(num_traj_states-1, num_controls)
#     print('Number of steps: ', num_traj_states)
#     return new_states, new_controls


###############################################################################
######################## FUNCTIONS CALLED BY MAIN #############################
###############################################################################
def get_rrtstar_optimal_trajectory(file_name, output_file_name, save_opt_path=False, plot_opt_path=False, save_opt_path_plot=False):
    """
    Finds the optimal trajectory as returned by RRT*
    inputs:
        file_name: name and path of RRT* pickle file
        output_file_name: name and path of output file
        save_opt_path: True --> save data, False --> don't save data
        plot_opt_path: True --> plot optimal path, False --> don't plot optimal path
        save_opt_path_plot: True --> save plot, False --> don't save plot
    """

    pathNodesList = pickle_import(file_name)

    # get RRT* nodes and inputs
    start_state, tree_states, tree_ctrls = get_sampled_traj_and_ctrls(pathNodesList)

    # get all states and controls for the optimal trajectory
    sampled_opt_traj_nodes_, full_opt_traj_states, full_opt_traj_ctrls = get_full_opt_traj_and_ctrls(pathNodesList)

    # save data
    if save_opt_path:
        save_data(full_opt_traj_states, full_opt_traj_ctrls, output_file_name)

    # plot data
    if plot_opt_path:
        plot_data(tree_states, sampled_opt_traj_nodes_, full_opt_traj_states, full_opt_traj_ctrls,
                  output_file_name, save_opt_path_plot, title_str='Original trajectory')

    return


def get_short_rrtstar_optimal_trajectory(file_name, output_file_name,
                                         save_opt_path=False, plot_opt_path=False, save_opt_path_plot=False):
    """
    Finds the optimal trajectory as returned by RRT* and shortens it (in time)
    inputs:
        file_name: name and path of RRT* pickle file
        output_file_name: name and path of output file
        v_max: maximum linear velocity
        omega_max: maximum angular velocity
        num_states: number of states
        num_controls: number of controls
        save_opt_path: True --> save data, False --> don't save data
        plot_opt_path: True --> plot optimal path, False --> don't plot optimal path
        save_opt_path_plot: True --> save plot, False --> don't save plot
    """

    pathNodesList = pickle_import(file_name)

    # get RRT* nodes and inputs
    start_state, tree_states, tree_ctrls = get_sampled_traj_and_ctrls(pathNodesList)

    # get all states and controls for the optimal trajectory
    sampled_opt_traj_nodes_, full_opt_traj_states, full_opt_traj_ctrls = get_full_opt_traj_and_ctrls(pathNodesList)

    # shorten trajectory
    shortened_traj, shortened_ctrls = shorten_traj(sampled_opt_traj_nodes_, full_opt_traj_states, full_opt_traj_ctrls)

    # save data
    if save_opt_path:
        save_data(shortened_traj[0:-1][:], shortened_ctrls, output_file_name)

    # plot data
    if plot_opt_path:
        plot_data(tree_states, sampled_opt_traj_nodes_, shortened_traj[0:-1][:], shortened_ctrls,
                  output_file_name, save_opt_path_plot, title_str='Shortened trajectory')

    return


def opt_and_short_traj(filename, save_path,
                       save_opt_path=True, plot_opt_path=True, save_opt_path_plot=True,
                       save_short_opt_path=True, plot_short_opt_path=True, save_short_opt_path_plot=True):
    """
    Combines both of the above
    Inputs:
        filename: just the RRT* pickle file name without the path
        save_path: path to RRT* pickle file
        v_max, omega_max, num_states, num_controls: max linear and angular velocity and number of states and controls
    """
    nodelist_string = "NodeListData"
    opt_traj_name = "OptTraj"
    shortened_traj_name = "OptTraj_short"

    new_filename_opt = filename.replace(nodelist_string, opt_traj_name)
    new_filename_short = filename.replace(nodelist_string, shortened_traj_name)
    filename = os.path.join(save_path, filename)
    new_filename_opt = os.path.join(save_path, new_filename_opt)
    new_filename_short = os.path.join(save_path, new_filename_short)

    get_rrtstar_optimal_trajectory(filename, new_filename_opt, save_opt_path=save_opt_path, plot_opt_path=plot_opt_path,
                                   save_opt_path_plot=save_opt_path_plot)
    get_short_rrtstar_optimal_trajectory(filename, new_filename_short,
                                         save_opt_path=save_short_opt_path, plot_opt_path=plot_short_opt_path, save_opt_path_plot=save_short_opt_path_plot)
    return


if __name__ == '__main__':
    from rans_rrtstar.filesearch import get_filename

    prefix = 'NodeListData'
    filename = get_filename(SAVEPATH, tstr='last', prefix=prefix)
    # filename = prefix + "_v2_0_" + "1627880310"

    opt_and_short_traj(filename, SAVEPATH)
