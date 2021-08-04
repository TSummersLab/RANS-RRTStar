#!/usr/bin/env python3
"""
Changelog:
New is v1_1:
- Fixes bug in heading angle when rewiring the tree

New is v1_0:
- Run DR-RRT* with unicycle dynamics for steering

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author:
Venkatraman Renganathan
Email:
vrengana@utdallas.edu
Date created: Mon Jan 13 12:10:03 2020
Contributions: DR-RRT* base code (structures, expanding functions, rewire functions, and plotting)
(C) Venkatraman Renganathan, 2019.

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
Github:
@The-SS
Contributions: Different steering function, updated steering and rewiring functions, adding features, RANS-RRT* updates

Author:
Benjamin Gravell
Email:
benjamin.gravell@utdallas.edu
GitHub:
@BenGravell
Contributions: NLP solver bugfixes, code reorganization

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script performs RRT or RRT* path planning using the unicycle dynamics for steering. Steering is achieved by solving
a nonlinear program (NLP) using Casadi.

"""

# Import all the required libraries
import math
import csv
import copy
import os
import time
from dataclasses import dataclass

import numpy as np
import numpy.random as npr
import numpy.linalg as la

import casadi as cas
from casadi import SX, mtimes, nlpsol

import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection


from rans_rrtstar import config

from rans_rrtstar.common.geometry import sample_rectangle, compute_L2_distance, saturate_node_with_L2
from rans_rrtstar.common.dynamics import DYN
from rans_rrtstar.common.ukf import UKF
from rans_rrtstar.common.plotting import plot_env

from utility.path_utility import create_directory
from utility.pickle_io import pickle_import, pickle_export


# Defining Global Variables (User chosen)
# See config file for defaults
SEED = config.SEED
NUMSAMPLES = config.NUMSAMPLES  # total number of samples
STEER_TIME = config.STEER_TIME  # Maximum Steering Time Horizon
ENVCONSTANT = config.ENVCONSTANT  # Environment Constant for computing search radius
DT = config.DT  # timestep between controls
GOALAREA = copy.deepcopy(config.GOALAREA)  # [xmin,xmax,ymin,ymax] Goal zone
ROBSTART = config.ROBSTART  # robot starting location (x,y)
RANDAREA = copy.deepcopy(config.RANDAREA)  # area sampled: [xmin,xmax,ymin,ymax], [-4.7, 4.7, -4.7, 4.7] good with 0 ROBRAD, limit:[-5,5,-5,5]
RRT = config.RRT  # True --> RRT, False --> RRT*
DRRRT = config.DRRRT  # True --> apply DR checks, False --> regular RRT
MAXDECENTREWIRE = config.MAXDECENTREWIRE  # maximum number of descendents to rewire
RANDNODES = config.RANDNODES  # false --> only 5 handpicked nodes for debugging
SATLIM = config.SATLIM  # saturation limit (random nodes sampled will be cropped down to meet this limit from the nearest node)
ROBRAD = config.ROBRAD  # radius of robot (added as padding to environment bounds and the obstacles
SBSP = config.SBSP  # Shrinking Ball Sampling Percentage (% nodes in ball to try to rewire) (100 --> all nodes rewired)
SBSPAT = config.SBSPAT  # SBSP Activation Threshold (min number of nodes needed to be in the shrinking ball for this to activate)
SAVEDATA = config.SAVEDATA  # True --> save data, False --> don't save data
SAVEPATH = config.SAVEPATH  # path to save data
OBSTACLELIST = copy.deepcopy(config.OBSTACLELIST)  # [ox,oy,wd,ht]
SIGMAW = config.SIGMAW  # Covariance of process noise
SIGMAV = config.SIGMAV  # Covariance of sensor noise (we don't have any for now)
CROSSCOR = config.CROSSCOR   # Cross Correlation between the two noises (none for now)
ALFA = config.ALFA  # risk bound
QHL = config.QHL  # Q matrix for quadratic cost
RHL = config.RHL  # R matrix for quadratic cost
# To edit robot speed go to `GetDynamics`
# To edit input and state quadratic cost matrices Q and R go to `SetUpSteeringLawParameters`


# Defining Global Variables (NOT User chosen)
from rans_rrtstar import file_version

FILEVERSION = file_version.FILEVERSION  # version of this file
SAVETIME = str(int(time.time()))  # Used in filename when saving data
if not RANDNODES:
    NUMSAMPLES = 5
create_directory(SAVEPATH)


@dataclass
class StartParams:
    start: list  # robot starting location [x, y]
    rand_area: list  # [xmin, xmax, ymin, ymax]
    goal_area: list  # [xmin, xmax, ymin, ymax]
    max_iter: int
    plot_frequency: int
    obstacle_list: list  # [[ox1, oy1, wd1, ht1], [ox2, oy2, wd2, ht2], ...]


@dataclass
class SteerSetParams:
    dt: float  # discretized time step
    f: ...  # CasADi Function of continuous-time dynamics
    N: int # Prediction horizon
    solver: ...  # CasADi solver object e.g. from nlpsol()
    argums: dict  # constraint argument dictionary
    num_states: int
    num_controls: int


def dummy_problem_ipopt():
    # This is just called to make the annoying IPOPT print banner show up before doing other stuff
    from casadi import SX, nlpsol
    x = SX.sym('x')
    nlp = {'x': x, 'f': x**2}
    settings = {'verbose': 0,
                'ipopt.print_level': 0,
                'print_time': 0}
    sol = nlpsol('sol', 'ipopt', nlp, settings)
    return sol()


class TrajNode:
    """
    Class Representing a steering law trajectory Node
    """
    def __init__(self, num_states, num_controls):
        self.X = np.zeros((num_states, 1))  # State Vector
        self.Sigma = np.zeros((num_states, num_states, 1))  # Covariance Matrix
        self.Ctrl = np.zeros(num_controls)  # Control at trajectory node


class TreeNode:
    """
    Class Representing a DR-RRT* Tree Node
    """
    def __init__(self, num_states, num_controls, num_traj_nodes):
        self.cost = 0.0  # Cost
        self.parent = None  # Index of the parent node
        self.means = np.zeros((num_traj_nodes, num_states, 1))  # Mean Sequence
        self.covars = np.zeros((num_traj_nodes, num_states, num_states))  # Covariance matrix sequence
        self.inputCommands = np.zeros((num_traj_nodes - 1, num_controls))  # Input Commands to steer from parent to the node itself

    def __eq__(self, other):
        costFlag = self.cost == other.cost
        parentFlag = self.parent == other.parent
        meansFlag = np.array_equal(self.means, other.means)
        covarsFlag = np.array_equal(self.covars, other.covars)

        return costFlag and parentFlag and meansFlag and covarsFlag


class Tree:
    """
    Class for DR-RRT* Planning
    """
    def __init__(self, start_param):
        """
        start_param: list of the following items:
            start   : Start Position [x,y]
            randArea: Ramdom Samping Area [xmin,xmax,ymin,ymax]
            goalArea: Goal Area [xmin,xmax,ymin,ymax]
            maxIter : Maximum # of iterations to run for constructing DR-RRT* Tree
        """

        # Add the Double Integrator Data
        self.iter = 0
        self.controlPenalty = 0.02
        self.plotFrequency = start_param.plot_frequency
        self.rand_area = start_param.rand_area
        self.goal_area = start_param.goal_area
        self.maxIter = start_param.max_iter
        self.obstacleList = start_param.obstacle_list
        self.num_states = DYN.n
        self.num_controls = DYN.m
        self.alfaThreshold = 0.01  # 0.05
        self.alfa = [self.alfaThreshold] * len(self.obstacleList)
        self.alfa = copy.deepcopy(ALFA)
        self.saturation_limit = SATLIM
        self.R = []
        self.Q = []
        self.dropped_samples = []
        self.S0 = np.array([[0.1, 0, 0],
                           [0, 0.1, 0],
                           [0, 0, 0.1]])
        # Define the covariances of process and sensor noises
        self.numOutputs = self.num_states
        self.SigmaW = SIGMAW  # 0.005 is good # Covariance of process noise
        self.SigmaV = SIGMAV  # Covariance of sensor noise
        self.CrossCor = CROSSCOR  # Cross Correlation between the two noises. # TODO Remove this later ?

        # Initialize DR-RRT* tree node with start coordinates
        self.initialize_tree(start_param.start)

    def initialize_tree(self, start):
        """
        Prepares DR-RRT* tree node with start coordinates & adds to node_list
        """

        # Create an instance of DR_RRTStar_Node class
        num_traj_nodes = 1
        self.start = TreeNode(self.num_states, self.num_controls, num_traj_nodes)

        for k in range(num_traj_nodes):
            self.start.means[k, 0, :] = start[0]
            self.start.means[k, 1, :] = start[1]
            self.start.covars[k, :, :] = self.S0
            # heading is already initialized to zero # TODO: try setting it to point to goal
            # no need to update the input since they are all zeros as initialized

            # Add the created start node to the node_list
        self.node_list = [self.start]

    def get_dynamics(self):
        # Define steer function variables
        dt = DYN.Ts  # discretized time step
        N = STEER_TIME  # Prediction horizon
        constraint_argdict = DYN.state_control_constraints()
        f = DYN.ctime_dynamics_cas()
        return dt, N, constraint_argdict, f

    def rand_free_checks(self, x, y):
        """
        Performs Collision Check For Random Sampled Point
        Inputs:
        x,y : Position data which has to be checked for collision
        Outputs:
        True if safe, False if collision
        """

        for ox, oy, wd, ht in self.obstacleList:
            if ox <= x <= ox + wd and oy <= y <= oy + ht:
                return False  # collision
        return True  # safe

    def get_random_point(self):
        """
        Returns a randomly sampled node from the obstacle free space
        """
        # Create a randNode as a tree object
        num_traj_nodes = 1
        randNode = TreeNode(self.num_states, self.num_controls, num_traj_nodes)

        # Initialize using the generated free points
        xFreePoints, yFreePoints, thetaFreePoints = self.freePoints
        randNode.means[-1, 0, :] = xFreePoints[self.iter]
        randNode.means[-1, 1, :] = yFreePoints[self.iter]
        randNode.means[-1, 2, :] = thetaFreePoints[self.iter]

        return randNode

    def get_free_random_points(self):
        xFreePoints = []
        yFreePoints = []
        thetaFreePoints = []

        if RANDNODES:
            # get 3% of the sampled points from the goal. The number is bounded within 2 and 10 nodes (2 < 3% < 10)
            # (added at the end of the list of nodes)
            points_in_goal = 0  # min(self.maxIter, min(max(2, int(3 / 100 * self.maxIter)), 10))
            for iter in range(self.maxIter - points_in_goal):
                # Sample uniformly around sample space
                searchFlag = True
                while searchFlag:
                    # initialize with a random position in space with orientation = 0
                    xPt, yPt = sample_rectangle(self.rand_area)
                    if self.rand_free_checks(xPt, yPt):
                        break
                xFreePoints.append(xPt)
                yFreePoints.append(yPt)
                thetaFreePoints.append(np.random.uniform(-np.pi, np.pi))
            for iter in range(points_in_goal):
                xPt, yPt = sample_rectangle(self.goal_area)
                thetaPt = np.random.uniform(-np.pi, np.pi)
                xFreePoints.append(xPt)
                yFreePoints.append(yPt)
                thetaFreePoints.append(thetaPt)
        else:
            # pre-chosen nodes (for debugging only)
            xFreePoints.append(-3)
            yFreePoints.append(-4)
            thetaFreePoints.append(0)
            xFreePoints.append(-1)
            yFreePoints.append(-4)
            thetaFreePoints.append(0)
            xFreePoints.append(0.5)
            yFreePoints.append(-4)
            thetaFreePoints.append(np.pi / 2)
            xFreePoints.append(0.5)
            yFreePoints.append(0.5)
            thetaFreePoints.append(np.pi / 2)
            xFreePoints.append(0)
            yFreePoints.append(-1.5)
            thetaFreePoints.append(np.pi / 4)

        self.freePoints = [xFreePoints, yFreePoints, thetaFreePoints]

    def GetAncestors(self, childNode):
        """
        Returns the complete list of ancestors for a given child Node
        """
        ancestornode_list = []
        while True:
            if childNode.parent is None:
                # It is root node - with no parents
                ancestornode_list.append(childNode)
                break
            elif childNode.parent is not None:
                ancestornode_list.append(self.node_list[childNode.parent])
                childNode = self.node_list[childNode.parent]
        return ancestornode_list

    def GetNearestListIndex(self, randNode):
        """
        Returns the index of the node in the tree that is closest to the randomly sampled node
        Input Parameters:
        randNode  : The randomly sampled node around which a nearest node in the DR-RRT* tree has to be returned
        """
        distanceList = []
        for node in self.node_list:
            distanceList.append(compute_L2_distance(node, randNode))
        return distanceList.index(min(distanceList))

    def PrepareTrajectory(self, meanValues, covarValues, inputCommands):
        """
        Prepares the trajectory as trajNode from steer function outputs

        Input Parameters:
        meanValues : List of mean values
        covarValues : List of covariance values
        inputCommands: List of input commands

        Output Parameters:
        xTrajs: List of TrajNodes
        """

        T = len(inputCommands)
        # Trajectory data as trajNode object for each steer time step
        xTrajs = [TrajNode(self.num_states, self.num_controls) for i in range(T + 1)]
        for k, xTraj in enumerate(xTrajs):
            xTraj.X = meanValues[k]
            xTraj.Sigma = covarValues[k]  # TODO: CHECK DIM'N/ELEMENT ACCESS
            if k < T:
                xTraj.Ctrl = inputCommands[k]
        return xTrajs

    def SetUpSteeringLawParameters(self):
        num_states, num_controls = self.num_states, self.num_controls

        # Get the dynamics specific data
        dt, N, argums, f = self.get_dynamics()

        # Define state and input cost matrices for solving the NLP
        Q = QHL
        R = RHL
        self.Q = Q
        self.R = R

        # Casadi SX trajectory variables/parameters for multiple shooting
        U = SX.sym('U', N, num_controls)  # N trajectory controls
        X = SX.sym('X', N + 1, num_states)  # N+1 trajectory states
        P = SX.sym('P', num_states + num_states)  # first and last states as independent parameters

        # Concatinate the decision variables (inputs and states)
        opt_variables = cas.vertcat(cas.reshape(U, -1, 1), cas.reshape(X, -1, 1))

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
        g.append(X[N, 0:2].T - P[3:5])  # constraint on final state

        # Set the nlp problem
        nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': cas.vertcat(*g)}

        # Set the nlp problem settings
        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 1,  # 4
                        'print_time': 0,
                        'verbose': 0,  # 1
                        'error_on_fail': 1}

        # Create a solver that uses IPOPT with above solver settings
        solver = nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
        self.SteerSetParams = SteerSetParams(dt, f, N, solver, argums, num_states, num_controls)

    def SetUpSteeringLawParametersWithFinalHeading(self):
        """
        Same as SetUpSteeringLawParameters but with all three states enforced at the end instead of only x-y states
        This is used when rewiring and updating descendants
        """
        num_states, num_controls = self.num_states, self.num_controls

        # Get the dynamics specific data
        dt, N, argums, f = self.get_dynamics()

        # Define state and input cost matrices for solving the NLP
        Q = QHL
        R = RHL
        self.Q = Q
        self.R = R

        # Casadi SX trajectory variables/parameters for multiple shooting
        U = SX.sym('U', N, num_controls)  # N trajectory controls
        X = SX.sym('X', N + 1, num_states)  # N+1 trajectory states
        P = SX.sym('P', num_states + num_states)  # first and last states as independent parameters

        # Concatinate the decision variables (inputs and states)
        opt_variables = cas.vertcat(cas.reshape(U, -1, 1), cas.reshape(X, -1, 1))

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
        nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': cas.vertcat(*g)}

        # Set the nlp problem settings
        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 0,  # 4
                        'print_time': 0,
                        'verbose': 0,  # 1
                        'error_on_fail': 1}

        # Create a solver that uses IPOPT with above solver settings
        solver = nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        self.SteerSetParams2 = SteerSetParams(dt, f, N, solver, argums, num_states, num_controls)

    def solveNLP(self, solver, argums, x0, xT, n_states, n_controls, N, T):
        """
        Solves the nonlinear steering problem using the solver from SetUpSteeringLawParameters
        Inputs:
            solver: Casadi NLP solver from SetUpSteeringLawParameters
            argums: argums with lbg, ubg, lbx, ubx
            x0, xT: initial and final states as (n_states)x1 ndarrays e.g. [[2.], [4.], [3.14]]
            n_states, n_controls: number of states and controls
            N: horizon
            T: time step
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

        # Initialize and solve NLP
        c_p = np.concatenate((x0, xT))  # start and goal state constraints
        init_decision_vars = np.concatenate((init_inputs.reshape(-1, 1), init_states.reshape(-1, 1)))
        try:
            res = solver(x0=init_decision_vars, p=c_p, lbg=argums['lbg'], lbx=argums['lbx'], ubg=argums['ubg'],
                         ubx=argums['ubx'])
        except:
            # raise Exception('NLP failed')
            print('NLP Failed')
            return None, None

        # Output series looks like [u0, u1, ..., uN, x0, x1, ..., xN+1]  #####[u0, x0, u1, x1, ...]
        casadi_result = res['x'].full()

        # Extract the control inputs and states
        u_casadi = casadi_result[:2 * N].reshape(n_controls, N).T  # (N, n_controls)
        x_casadi = casadi_result[2 * N:].reshape(n_states, N + 1).T  # (N+1, n_states)

        return x_casadi, u_casadi

    def nonlinsteer(self, steerParams):
        """
        Use created solver from `SetUpSteeringLawParameters` to solve NLP using `solveNLP` then rearrange the results
        """

        # Unbox the input parameters
        fromNode = steerParams["fromNode"]
        toNode = steerParams["toNode"]
        SteerSetParams = steerParams["Params"]

        # Unwrap the variables needed for simulation
        N = SteerSetParams.N  # Steering horizon
        solver = SteerSetParams.solver  # NLP IPOPT Solver object
        argums = SteerSetParams.argums  # NLP solver arguments
        num_states = SteerSetParams.num_states  # Number of states
        num_controls = SteerSetParams.num_controls  # Number of controls

        # Feed the source and destination parameter values
        x0 = fromNode.means[-1, :, :]  # source
        xGoal = toNode.means[-1, :, :]  # destination

        [x_casadi, u_casadi] = self.solveNLP(solver, argums, x0, xGoal, num_states, num_controls, N, DT)

        if x_casadi is None or u_casadi is None:
            # NLP problem failed to find a solution
            steerResult = False
            steerOutput = {"means": [],
                           "covars": [],
                           "cost": [],
                           "steerResult": steerResult,
                           "inputCommands": []}
            return steerOutput
        else:
            # Steering is successful
            steerResult = True
            xHist = []
            uHist = []

            # add states/controls to xHist/uHist
            for i in range(len(u_casadi)):
                xHist.append(x_casadi[i, :].reshape(num_states, 1))
                uHist.append(u_casadi[i, :])
            xHist.append(x_casadi[-1, :].reshape(num_states, 1))

            # compute steering cost
            Q = self.Q
            R = self.R
            QT = self.Q
            steeringCost = 0
            xGoal3 = xGoal.reshape(num_states)  # Goal node we tried to steer to
            for i in range(len(uHist)):
                cost_i = 0
                # # state cost
                # state_i = copy.copy(xHist[i])
                # state_i = state_i.reshape(num_states) # subtract the desired location
                # cost_i += (state_i-xGoal3).dot(Q).dot(state_i-xGoal3)
                # control effort cost
                ctrl_i = copy.copy(uHist[i])
                ctrl_i = ctrl_i.reshape(num_controls)
                cost_i += ctrl_i.dot(R).dot(ctrl_i)
                # update steering cost
                steeringCost += cost_i
            # add cost on final state relative to goal. should always be zero
            state_i = copy.copy(xHist[i + 1])
            state_i = state_i.reshape(num_states)  # subtract the desired location
            cost_i = (state_i - xGoal3).dot(QT).dot(state_i - xGoal3)
            steeringCost += cost_i

            # Find covariances
            if DRRRT:
                covarHist = self.ukfCovars(xHist, uHist, steerParams)
            else:
                covarHist = [np.zeros([num_states, num_states])] * (N + 1)

            # Prepare output dictionary
            steerOutput = {"means": xHist,
                           "covars": covarHist,
                           "cost": steeringCost,
                           "steerResult": steerResult,
                           "inputCommands": uHist}
            return steerOutput

    def ukfCovars(self, xHist, uHist, steerParams):
        # Unbox the input parameters
        fromNode = steerParams["fromNode"]
        SteerSetParams = steerParams["Params"]

        # Unwrap the variables needed for simulation
        N = SteerSetParams.N  # Steering horizon
        num_states = self.num_states  # Number of states

        ukf_params = {}
        ukf_params["n_x"] = self.num_states
        ukf_params["n_o"] = self.numOutputs
        ukf_params["SigmaW"] = self.SigmaW
        ukf_params["SigmaV"] = self.SigmaV
        ukf_params["CrossCor"] = self.CrossCor # TODO: DEFINED THIS IS __init__
        ukf_params["dT"] = DT

        # Find covariances
        SigmaE = fromNode.covars[-1, :, :]  # covariance at initial/from node
        covarHist = [SigmaE]
        for k in range(0, N): # TODO: is this up to N or N-1
            x_hat = xHist[k] # TODO: k-th state?
            u_k = uHist[k] # TODO: k-th control? ALSO CHECK DIM'N
            y_k = xHist[k+1] # (we assume perfect full state feedback so y = x) TODO: k+1-th measurement = k+1 state?

            ukf_params["x_hat"] = x_hat
            ukf_params["u_k"] = u_k
            ukf_params["SigmaE"] = SigmaE
            ukf_params["y_k"] = y_k

            ukf_estimator = UKF()  # initialize the state estimator
            estimator_output = ukf_estimator.estimate(ukf_params)  # get the estimates
            x_hat = np.squeeze(estimator_output["x_hat"])  # Unbox the state (this is the same as the input x_hat = xHist so we don't need it)
            SigmaE = estimator_output["SigmaE"]  # Unbox the covariance
            covarHist.append(SigmaE.reshape(num_states, num_states))

        return covarHist

    def PerformCollisionCheck(self, xTrajs):
        """
        Performs point-obstacle & line-obstacle check in distributionally robust fashion.
        Input Parameters:
        xTrajs: collection of means of points along the steered trajectory
        Outputs:
        Ture if safe, Flase if collision
        """
        for k, xTraj in enumerate(xTrajs):
            if k != 0:
                # DR - Point-Obstacle Collision Check
                # collisionFreeFlag = True: Safe Trajectory and False: Unsafe Trajectory
                drCollisionFreeFlag = self.DRCollisionCheck(xTrajs[k])
                if not drCollisionFreeFlag:
                    # print('Point-Obstacle Collision Detected :::::::::')
                    return False
                    # # DR - Line-Obstacle Collision Check via LTL specifications
                drSTLCollisionFreeFlag = self.DRSTLCollisionCheck(xTrajs[k - 1], xTrajs[k])
                if not drSTLCollisionFreeFlag:
                    # print('Line-Obstacle Collision Detected ---------')
                    return False
                    # If everything is fine, return True
        return True

    def DRCollisionCheck(self, trajNode): #TODO: CHECK LATER
        """
        Performs Collision Check Using Deterministic Tightening of DR Chance
        Constraint and enforces constraints to be satisfied in two successive
        time steps to avoid jumping over obstacles between the waypoints.
        Inputs:
        trajNode  : Node containing data to be checked for collision
        Outputs:
        True if safe, False if collision
        """
        # Define the direction arrays
        xDir = np.array([1, 0, 0])
        yDir = np.array([0, 1, 0])

        # Initialize the flag to be true
        drCollisionFreeFlag = True

        for alpha, (ox, oy, wd, ht) in zip(self.alfa, self.obstacleList):
            # Check if the trajNode is inside the bloated obstacle (left and right and bottom and top)
            # TODO: CHECK HOW THE 2-NORM TERM IS FOUND LATER
            Delta = math.sqrt((1-alpha)/alpha)
            # print('padding: ', Delta*math.sqrt(xDir.T @ trajNode.Sigma @ xDir))
            if trajNode.X[0] >= (ox - Delta*math.sqrt(xDir.T @ trajNode.Sigma @ xDir)) and \
                trajNode.X[0] <= (ox + wd + Delta*math.sqrt(xDir.T @ trajNode.Sigma @ xDir) ) and \
                trajNode.X[1] <= (oy - Delta*math.sqrt(yDir.T @ trajNode.Sigma @ yDir)) and \
                trajNode.X[1] >= (oy + ht + Delta*math.sqrt(yDir.T @ trajNode.Sigma @ yDir)):
                # collision has occured, so return false
                drCollisionFreeFlag = False
                return drCollisionFreeFlag

        return drCollisionFreeFlag  # safe, so true

    def DRSTLCollisionCheck(self, firstNode, secondNode): # TODO: CHECK LATER
        """
        Performs Collision Check Using Deterministic Tightening of DR Chance
        Constraint and enforces constraints to be satisfied in two successive
        time steps to avoid jumping over obstacles between the waypoints.
        Input Parameters:
        firstNode  : 1st Node containing data to be checked for collision
        secondNode : 2nd Node containing data to be checked for collision
        """
        xDir = np.array([1, 0, 0])
        yDir = np.array([0, 1, 0])

        # Get the coordinates of the Trajectory line connecting two points
        x1 = firstNode.X[0]
        y1 = firstNode.X[1]
        x2 = secondNode.X[0]
        y2 = secondNode.X[1]

        itr = 0
        for alpha, (ox, oy, wd, ht) in zip(self.alfa, self.obstacleList):
            itr += 1
            Delta = math.sqrt((1 - alpha) / alpha)
            # Prepare bloated version of min and max x,y positions of obstacle
            minX = ox - Delta * math.sqrt(xDir.T @ secondNode.Sigma @ xDir)
            minY = oy - Delta * math.sqrt(yDir.T @ secondNode.Sigma @ yDir)
            maxX = ox + wd + Delta * math.sqrt(xDir.T @ secondNode.Sigma @ xDir)
            maxY = oy + ht + Delta * math.sqrt(yDir.T @ secondNode.Sigma @ yDir)

            # Condition for Line to be Completely outside the rectangle
            if (x1 <= minX and x2 <= minX or
                    y1 <= minY and y2 <= minY or
                    x1 >= maxX and x2 >= maxX or
                    y1 >= maxY and y2 >= maxY):
                continue

            # Calculate the slope of the line
            lineSlope = (y2 - y1) / (x2 - x1)

            # Connect with a line to other point and check if it lies inside
            yPoint1 = lineSlope * (minX - x1) + y1
            yPoint2 = lineSlope * (maxX - x1) + y1
            xPoint1 = (minY - y1) / lineSlope + x1
            xPoint2 = (maxY - y1) / lineSlope + x1

            if (yPoint1 > minY and yPoint1 < maxY or
                    yPoint2 > minY and yPoint2 < maxY or
                    xPoint1 > minX and xPoint1 < maxX or
                    xPoint2 > minX and xPoint2 < maxX):
                # print('COLLISION DETECTED !!!!!!!!!!!!!!!!!!!!!')
                # print('Obstacle number:', itr)
                # print('Obstacle: ', [ox, oy, wd, ht])
                # print('x1,y1:',x1,y1)
                # print('x2,y2:', x2, y2)
                # print('minX', minX)
                # print('minY', minY)
                # print('maxX', maxX)
                # print('maxY', maxY)
                return False

        return True  # Collision Free - No intersection

    def PrepareMinNode(self, nearestIndex, xTrajs, trajCost):
        """
        Prepares and returns the randNode to be added to the DR-RRT* tree
        Input Parameters:
        nearestIndex : Index of the nearestNode in the DR-RRT* tree
        xTrajs       : Trajectory data containing the sequence of means
        trajCost     : Cost to Steer from nearestNode to randNode
        inputCommands: Input commands needed to steer from nearestNode to randNode
        """
        # Convert trajNode to DR-RRT* Tree Node
        num_traj_nodes = len(xTrajs)
        minNode = TreeNode(self.num_states, self.num_controls, num_traj_nodes)
        # Associate the DR-RRT* node with sequence of means
        for k, xTraj in enumerate(xTrajs):
            minNode.means[k, :, :] = xTraj.X
            minNode.covars[k, :, :] = xTraj.Sigma
            if k < num_traj_nodes - 1:
                minNode.inputCommands[k, :] = xTraj.Ctrl
        minNode.cost = self.node_list[nearestIndex].cost + trajCost
        # Associate MinNode's parent as NearestNode
        minNode.parent = nearestIndex
        return minNode

    def FindNearNodeIndices(self, randNode):
        """
        Returns indices of all nodes that are closer to randNode within a specified radius
        Input Parameters:
        randNode : Node around which the nearest indices have to be selected
        """
        totalNodes = len(self.node_list)
        searchRadius = ENVCONSTANT * ((math.log(totalNodes + 1) / totalNodes + 1)) ** (1 / 2)
        distanceList = []
        for node in self.node_list:
            distanceList.append(compute_L2_distance(node, randNode))
        nearIndices = [distanceList.index(dist) for dist in distanceList if dist <= searchRadius ** 2]
        return nearIndices

    def ConnectViaMinimumCostPath(self, nearestIndex, nearIndices, randNode, minNode):  # TODO: DOUBLE CHECK THAT NO OTHER CHANGES ARE REQUIRED
        """
        Chooses the minimum cost path by selecting the correct parent
        Input Parameters:
        nearestIndex : Index of DR-RRT* Node that is nearest to the randomNode
        nearIndices  : Indices of the nodes that are nearest to the randNode
        randNode     : Randomly sampled node
        minNode      : randNode with minimum cost sequence to connect as of now
        """
        # If the queried node is a root node, return the same node
        if not nearIndices:
            return minNode

        # If there are other nearby nodes, loop through them
        for j, nearIndex in enumerate(nearIndices):
            # Looping except nearestNode - Uses the overwritten equality check function
            if self.node_list[nearIndex] == self.node_list[nearestIndex]:
                continue
            # if nearIndex == nearestIndex: # TODO: Try this instead
            #     continue

            # Try steering from nearNode to randNodeand get the trajectory
            success_steer, temp_minNode = self.SteerAndGetMinNode(from_idx=nearIndex, to_node=randNode)  # TODO: EDIT THIS FUNCTION

            # If steering failed, move on
            if not success_steer:
                continue
            # If steering succeeds, check if the new steering cost is less than the current minNode steering cost
            if temp_minNode.cost < minNode.cost:
                # If lower cost, update minNode
                minNode = copy.copy(temp_minNode)

        return minNode

    def rewire(self, nearIndices, minNode):  # TODO: DOUBLE CHECK - EDITED
        """
        Rewires the DR-RRT* Tree using Minimum cost path found
        Input Parameters:
        nearIndices : Indices of the nodes that are nearest to the randomNode
        minNode     : randNode with minimum cost sequence to connect as of now
        """
        # Get all ancestors of minNode
        minNodeAncestors = self.GetAncestors(minNode)
        for j, nearIndex in enumerate(nearIndices):
            # Avoid looping all ancestors of minNode
            if np.any([self.node_list[nearIndex] == minNodeAncestor for minNodeAncestor in minNodeAncestors]):
                continue

            if len(nearIndices) > SBSPAT and npr.rand() < 1 - (SBSP / 100):
                continue

            # steer from the minNode to the nodes around it with nearIndex and find the trajectory
            success_steer, xTrajs, sequenceCost = self.SteerAndGenerateTrajAndCostWithFinalHeading(from_node=minNode, to_idx=nearIndex)

            # If steering fails, move on
            if not success_steer:
                continue

            connectCost = minNode.cost + sequenceCost
            # Proceed only if J[x_min] + del*J(sigma,pi) < J[X_near]
            if connectCost < self.node_list[nearIndex].cost:
                self.node_list[nearIndex].parent = len(self.node_list) - 1
                self.node_list[nearIndex].cost = connectCost
                # prepare the means and inputs
                num_traj_nodes = len(xTrajs)
                meanSequence = np.zeros((num_traj_nodes, self.num_states, 1))  # Mean Sequence
                covarSequence = np.zeros((num_traj_nodes, self.num_states, self.num_states))  # Covar Sequence
                inputCtrlSequence = np.zeros((num_traj_nodes - 1, self.num_controls))  # Input Sequence
                for k, xTraj in enumerate(xTrajs):
                    meanSequence[k, :, :] = xTraj.X
                    covarSequence[k, :, :] = xTraj.Sigma
                    if k < num_traj_nodes - 1:
                        inputCtrlSequence[k, :] = xTraj.Ctrl
                # overwrite the mean and inputs sequences in the nearby node
                self.node_list[nearIndex].means = meanSequence  # add the means from xTrajs
                self.node_list[nearIndex].covars = covarSequence  # add the covariances from xTrajs
                self.node_list[nearIndex].inputCommands = inputCtrlSequence  # add the controls from xTrajs
                # Update the children of nearNode about the change in cost
                rewire_count = 0
                self.UpdateDescendantsCost(self.node_list[nearIndex], rewire_count)

    def UpdateDescendantsCost(self, newNode, rewire_count):  # TODO: DOUBLE CHECK - EDITED
        """
        Updates the cost of all children nodes of newNode
        Input Parameter:
        newNode: Node whose children's costs have to be updated
        """
        rewire_count += 1
        if rewire_count > MAXDECENTREWIRE:
            return
        # Record the index of the newNode
        newNodeIndex = self.node_list.index(newNode)
        # Loop through the node_list to find the children of newNode
        for childNode in self.node_list[newNodeIndex:]:
            # Ignore Root node and all ancestors of newNode - Just additional check
            if childNode.parent is None or childNode.parent < newNodeIndex:
                continue
            if childNode.parent == newNodeIndex:
                success_steer, xTrajs, trajCost = self.SteerAndGenerateTrajAndCostWithFinalHeading(from_idx=newNodeIndex,
                                                                                                    to_node=childNode)
                if not success_steer:
                    continue

                childNode.cost = newNode.cost + trajCost

                # prepare the means and inputs
                num_traj_nodes = len(xTrajs)
                meanSequence = np.zeros((num_traj_nodes, self.num_states, 1))  # Mean Sequence
                covarSequence = np.zeros((num_traj_nodes, self.num_states, self.num_states))  # Covar Sequence
                inputCtrlSequence = np.zeros((num_traj_nodes - 1, self.num_controls))  # Input Sequence
                for k, xTraj in enumerate(xTrajs):
                    meanSequence[k, :, :] = xTraj.X
                    covarSequence[k, :, :] = xTraj.Sigma
                    if k < num_traj_nodes - 1:
                        inputCtrlSequence[k, :] = xTraj.Ctrl
                # overwrite the mean and inputs sequences in the nearby node
                childNode.means = meanSequence
                childNode.covars = covarSequence
                childNode.inputCommands = inputCtrlSequence
                # Get one more level deeper
                self.UpdateDescendantsCost(childNode, rewire_count)

    def GetGoalNodeIndex(self):
        """
        Get incides of all RRT nodes in the goal region
        Inputs :
        NONE
        Outputs:
        goalNodeIndex: list of indices of the RRT nodes (type: python list)
        """
        # Get the indices of all nodes in the goal area
        goalIndices = []
        for node in self.node_list:
            if (node.means[-1, 0, :] >= self.xmingoal and node.means[-1, 1, :] >= self.ymingoal and
                node.means[-1, 0, :] <= self.xmaxgoal and node.means[-1, 0, :] <= self.ymaxgoal):  # TODO is this a typo??? node.means[-1, 0, :] <= self.ymaxgoal should be node.means[-1, 1, :] <= self.ymaxgoal  ???
                goalIndices.append(self.node_list.index(node))

        # Select a random node from the goal area
        goalNodeIndex = np.random.choice(goalIndices)

        return goalNodeIndex

    def GenerateSamplePath(self, goalIndex):
        """
        Generate a list of RRT nodes from the root to a node with index goalIndex
        Inputs:
        goalIndex: index of RRT node which is set as the goal
        Outputs:
        pathNodesList: list of RRT nodes from root node to goal node (type: python list (element type: DR_RRTStar_Node)) # TODO: check this
        """
        pathNodesList = [self.node_list[goalIndex]]

        # Loop until the root node (whose parent is None) is reached
        while self.node_list[goalIndex].parent is not None:
            # Set the index to its parent
            goalIndex = self.node_list[goalIndex].parent
            # Append the parent node to the pathnode_list
            pathNodesList.append(self.node_list[goalIndex])

        # Finally append the path with root node
        pathNodesList.append(self.node_list[0])

        return pathNodesList

    def SteerAndGenerateTrajAndCost(self, from_idx=None, from_node=None, to_idx=None, to_node=None):
        """
        Apply steering function to navigate from a starting node in the tree to a given node
        Perform a collision check
        Return the trajectory and cost between the two nodes
        Inputs:
        from_idx : index of node in the tree to navigate from
        to_node  : node to be added (DR_RRTStar_Node)
        Outputs:
        - Steering success flag (Type: bool)
        - Prepared trajectory (xTrajs) returned by PrepareTrajectory (type: # TODO: fill this)
        - Trajectory cost (type: float # TODO: CHECK THIS)
        The three outputs can have one of two options
        - True, xTrajs, trajCost: if steering succeeds (True), a trajectory is prepared (xTrajs); its cost is trajCost
        - return False, [], 0: if steering fails (False), the other parameters are set to bad values [] and 0 # TODO: consider replacing 0 with inf
        """
        # Steer from nearestNode to the randomNode using LQG Control
        # Returns a list of node points along the trajectory and cost
        # Box the steer parameters
        if from_idx == None:  # from index not given
            from_node_chosen = from_node
        else:  # from index given
            from_node_chosen = self.node_list[from_idx]

        if to_idx == None:  # to index not given
            to_node_chosen = to_node
        else:  # to index given
            to_node_chosen = self.node_list[to_idx]

        steerParams = {"fromNode": from_node_chosen,
                       "toNode": to_node_chosen,
                       "Params": self.SteerSetParams}

        steerOutput = self.nonlinsteer(steerParams)

        # Unbox the steer function output
        meanValues = steerOutput["means"]
        covarValues = steerOutput["covars"]
        trajCost = steerOutput["cost"]
        steerResult = steerOutput["steerResult"]
        inputCommands = steerOutput["inputCommands"]

        # If the steering law fails, force next iteration with different random sample
        if steerResult == False:
            # print('NLP Steering Failed XXXXXXXXX')
            return False, [], 0

        # Proceed only if the steering law succeeds
        # Prepare the trajectory
        xTrajs = self.PrepareTrajectory(meanValues, covarValues, inputCommands)

        # Check for Distributionally Robust Feasibility of the whole trajectory
        collisionFreeFlag = self.PerformCollisionCheck(xTrajs)

        # If a collision was detected, stop and move on
        if not collisionFreeFlag:
            # print('DR Collision Detected @@@@@@@@@')
            return False, [], 0

        return True, xTrajs, trajCost

    def SteerAndGenerateTrajAndCostWithFinalHeading(self, from_idx=None, from_node=None, to_idx=None, to_node=None):
        """
        Same as SteerAndGenerateTrajAndCost but uses the steering params, and hence the steering law, with the heading enforced to match the set value
        """
        # Steer from nearestNode to the randomNode using LQG Control
        # Returns a list of node points along the trajectory and cost
        # Box the steer parameters
        if from_idx == None:  # from index not given
            from_node_chosen = from_node
        else:  # from index given
            from_node_chosen = self.node_list[from_idx]

        if to_idx == None:  # to index not given
            to_node_chosen = to_node
        else:  # to index given
            to_node_chosen = self.node_list[to_idx]

        steerParams = {"fromNode": from_node_chosen,
                       "toNode": to_node_chosen,
                       "Params": self.SteerSetParams2}
        steerOutput = self.nonlinsteer(steerParams)

        # Unbox the steer function output
        meanValues = steerOutput["means"]
        covarValues = steerOutput["covars"]
        trajCost = steerOutput["cost"]
        steerResult = steerOutput["steerResult"]
        inputCommands = steerOutput["inputCommands"]

        # If the steering law fails, force next iteration with different random sample
        if steerResult == False:
            # print('NLP Steering Failed XXXXXXXXX')
            return False, [], 0

        # Proceed only if the steering law succeeds
        # Prepare the trajectory
        xTrajs = self.PrepareTrajectory(meanValues, covarValues, inputCommands)

        # Check for Distributionally Robust Feasibility of the whole trajectory
        collisionFreeFlag = self.PerformCollisionCheck(xTrajs)

        # If a collision was detected, stop and move on
        if not collisionFreeFlag:
            # print('DR Collision Detected @@@@@@@@@')
            return False, [], 0

        return True, xTrajs, trajCost

    def SteerAndGetMinNode(self, from_idx=None, from_node=None, to_idx=None, to_node=None):
        # steer and find the trajectory and trajectory cost
        success_steer, xTrajs, trajCost = self.SteerAndGenerateTrajAndCost(from_idx=from_idx, to_node=to_node)

        # If steering failed, stop
        if not success_steer:
            return False, []

        # If steering succeeds
        # Create minNode with trajectory data & do not add to the tree for the time being
        minNode = self.PrepareMinNode(from_idx, xTrajs, trajCost)

        return True, minNode

    def ExpandTree(self):
        """
        Subroutine that grows DR-RRT* Tree
        """

        # Prepare And Load The Steering Law Parameters
        self.SetUpSteeringLawParameters()
        self.SetUpSteeringLawParametersWithFinalHeading()

        # Generate maxIter number of free points in search space
        t1 = time.time()
        self.get_free_random_points()
        t2 = time.time()
        print('Finished Generating Free Points !!! Time elapsed: ', t2 - t1)

        num_steer_fail = 0  # number of nodes that fail to steer
        # Iterate over the maximum allowable number of nodes
        for iter in range(self.maxIter):
            print('Iteration Number %6d / %6d' % (iter+1, self.maxIter))
            self.iter = iter

            # Get a random feasible point in the space as a DR-RRT* Tree node
            randNode = self.get_random_point()
            # print('Trying randNode: ', randNode.means[-1,0,:][0], randNode.means[-1,1,:][0])

            # Get index of best DR-RRT* Tree node that is nearest to the random node
            nearestIndex = self.GetNearestListIndex(randNode)

            # Saturate randNode
            randNode = saturate_node_with_L2(self.node_list[nearestIndex], randNode)
            xFreePoints, yFreePoints, thetaFreePoints = self.freePoints
            xFreePoints[self.iter] = randNode.means[-1, 0, :]
            yFreePoints[self.iter] = randNode.means[-1, 1, :]
            thetaFreePoints[self.iter] = randNode.means[-1, 2, :]
            if not self.rand_free_checks(randNode.means[-1, 0, :], randNode.means[-1, 1, :]):
                self.dropped_samples.append(iter)
                continue

            success_steer, minNode = self.SteerAndGetMinNode(from_idx=nearestIndex, to_node=randNode)

            if not success_steer:
                num_steer_fail += 1
                continue

            if RRT:
                self.node_list.append(minNode)
            else:
                # Get all the nodes in the DR-RRT* Tree that are closer to the randomNode within a specified search radius
                nearIndices = self.FindNearNodeIndices(randNode)
                # Choose the minimum cost path to connect the random node
                minNode = self.ConnectViaMinimumCostPath(nearestIndex, nearIndices, randNode, minNode)
                # Add the minNode to the DR-RRT* Tree
                self.node_list.append(minNode)
                # Rewire the tree with newly added minNode
                self.rewire(nearIndices, minNode)

        print('~~~~~~~~~~~~~~~~~~ DONE TREE EXPANSION ~~~~~~~~~~~~~~~~~~')
        print('Number of Nodes Dropped (Saturation Fail): ', len(self.dropped_samples))
        print('Number of Nodes Failed (Steer Fail): ', num_steer_fail)
        return self.node_list


def plot_saved_data(pathNodesList, filename,
                    tree=None,
                    plot_tree_node_centers=True,
                    plot_rejected_nodes=False,
                    show_start=True,
                    plot_dr_check_ellipse=True):
    # Initialize figure
    fig, ax = plt.subplots(figsize=[9, 9])

    # Plot the environment
    ax = plot_env(ax)

    # Plot the start point
    if show_start:
        start = pathNodesList[0]
        ax.scatter(start.means[-1, 0, :], start.means[-1, 1, :], s=200, c='w', edgecolor='k', linewidths=2, marker='^',
                   label='Start', zorder=200)

    # Plot rejected nodes
    if plot_rejected_nodes:
        if tree is None:
            raise Exception('Cannot plot rejected nodes because tree object was not furnished!')
        else:
            x_sampled = []
            y_sampled = []
            xFreePoints, yFreePoints, thetaFreePoints = tree.freePoints
            for i in range(len(xFreePoints)):
                if not i in tree.dropped_samples:  # skip nodes that became infeasible after saturation
                    x_sampled.append(xFreePoints[i])
                    y_sampled.append(yFreePoints[i])
            plt.plot(x_sampled, y_sampled,'o', color='red', markersize=3)

    # Plot sampled nodes that were added to the tree
    x_added = []
    y_added = []
    for k, node in enumerate(pathNodesList):
        x_added.append(node.means[-1, 0, :][0])
        y_added.append(node.means[-1, 1, :][0])
    if plot_tree_node_centers:
        plt.plot(x_added, y_added, 'o', color='black', markersize=3)

    # Prepare to plot trajectory lines and safety ellipses
    xValues = []
    yValues = []
    widthValues = []
    heightValues = []
    angleValues = []
    lineObjects = []

    alpha = np.array(ALFA, float)
    delta = (1 - alpha) / alpha
    delta = delta**0.5
    eps = 0.0001
    all_deltas_same = all(delta[0] - eps <= elt <= delta[0] + eps for elt in list(delta))
    if not all_deltas_same:
        # if not all risk bounds are the same, plotting the dr padding on the robot doesn't make sense
        # (different paddings for every obstacle)
        plot_dr_check_ellipse = False

    # Create the ellipse geometry and trajectory lines
    for ellipseNode in pathNodesList:
        if ellipseNode is not None and ellipseNode.parent is not None:
            ellNodeShape = ellipseNode.means.shape
            xPlotValues = []
            yPlotValues = []
            # Prepare the trajectory x and y vectors and plot them
            for k in range(ellNodeShape[0]):
                xPlotValues.append(ellipseNode.means[k, 0, 0])
                yPlotValues.append(ellipseNode.means[k, 1, 0])
            # Plot the trajectory lines
            lx, = ax.plot(xPlotValues,
                          yPlotValues,
                          # color='#636D97',
                          # linewidth=1.0, alpha=0.9)
                          color='#0078f0',
                          linewidth=1.0,
                          alpha=0.9)
            lineObjects.append(lx)

            if plot_dr_check_ellipse:  # plot dr_coll check ellipse
                # Plot only the last ellipse in the trajectory
                alfa = math.atan2(ellipseNode.means[-1, 1, 0],
                                  ellipseNode.means[-1, 0, 0])
                elcovar = np.asarray(ellipseNode.covars[-1, :, :])  # covariance
                # plot node dr-check
                xValues.append(ellipseNode.means[-1, 0, 0])
                yValues.append(ellipseNode.means[-1, 1, 0])
                xDir = np.array([1, 0, 0])
                yDir = np.array([0, 1, 0])
                Delta = delta[-1]  # use environment level of padding
                major_ax_len = (Delta * math.sqrt(
                    xDir.T @ elcovar @ xDir)) * 2  # (.) * 2 <--  because we want width of ellipse
                minor_ax_len = (Delta * math.sqrt(
                    yDir.T @ elcovar @ yDir)) * 2  # --> padding in right and left directions added
                widthValues.append(major_ax_len)
                heightValues.append(minor_ax_len)
                angleValues.append(alfa * 360)
            else:  # do not plot dr_coll check
                alfa = math.atan2(ellipseNode.means[-1, 1, 0],
                                  ellipseNode.means[-1, 0, 0])
                xValues.append(ellipseNode.means[-1, 0, 0])
                yValues.append(ellipseNode.means[-1, 1, 0])
                widthValues.append(0)
                heightValues.append(0)
                angleValues.append(alfa * 360)

    # Plot the safe ellipses
    XY = np.column_stack((xValues, yValues))
    ec = EllipseCollection(widthValues,
                           heightValues,
                           angleValues,
                           units='x',
                           offsets=XY,
                           facecolors="#C59434",
                           transOffset=ax.transData,
                           alpha=0.25)
    ax.add_collection(ec)

    axis_limits = [-5.2, 5.2, -5.2, 5.2]
    ax.axis('equal')
    ax.axis(axis_limits)
    ax.axis('off')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.autoscale(False)

    plt.pause(1.0001)
    if SAVEDATA:
        plot_filename = filename.replace('NodeListData', "plot_tree")
        plot_name = plot_filename + '.png'
        plot_name = os.path.join(SAVEPATH, plot_name)
        plt.savefig(plot_name)

    plt.show()
    return fig, ax


def make_start_parameters():
    start = ROBSTART
    rand_area = copy.copy(RANDAREA)  # [xmin,xmax,ymin,ymax]
    goal_area = copy.copy(GOALAREA)  # [xmin,xmax,ymin,ymax]
    maxIter = NUMSAMPLES
    plotFrequency = 1
    # Obstacle Location Format [ox,oy,wd,ht]:
    # ox, oy specifies the bottom left corner of rectangle with width: wd and height: ht
    obstacleList = copy.deepcopy(OBSTACLELIST)

    # environment rectangle bottom left and top right corners
    xmin = rand_area[0]
    xmax = rand_area[1]
    ymin = rand_area[2]
    ymax = rand_area[3]
    # thickness of env edges (doesn't matter much, anything > 0  works)
    thickness = 0.1
    # original environment area - width and height
    width = xmax - xmin
    height = ymax - ymin

    # top, bottom, right, and left rectangles for the env edges
    env_bottom = [xmin-thickness, ymin-thickness, width+2*thickness, thickness]
    env_top = [xmin-thickness, ymax, width+2*thickness, thickness]
    env_right = [xmax, ymin-thickness, thickness, height+2*thickness]
    env_left = [xmin-thickness, ymin-thickness, thickness, height+2*thickness]

    obstacleList.append(env_bottom)
    obstacleList.append(env_top)
    obstacleList.append(env_right)
    obstacleList.append(env_left)

    # TODO: add env to obstacleList. Might have to remove rand_area padding

    # add padding to rand_area bounds:
    rand_area[0] += ROBRAD  # increase minimum x by robot radius
    rand_area[1] -= ROBRAD  # decrease maximum x by robot radius
    rand_area[2] += ROBRAD  # increase minimum y by robot radius
    rand_area[3] -= ROBRAD  # decrease maximum y by robot radius

    # add enough padding for obstacles for robot radius
    for obs in obstacleList:
        obs[0] -= ROBRAD  # decrease bottom left corner along x direction by robot radius
        obs[1] -= ROBRAD  # decrease bottom left corner along y direction by robot radius
        obs[2] += (2 * ROBRAD)  # increase width of obstacle by robot diameter
        obs[3] += (2 * ROBRAD)  # increase height of obstacle by robot diameter

    return StartParams(start, rand_area, goal_area, maxIter, plotFrequency, obstacleList)


def write_pathnodes_csv(pathNodesList, filename, print_pathnodes=False):
    path_out = os.path.join(SAVEPATH, filename) + '.csv'
    with open(path_out, mode='w') as csv_saver:
        csv_writer = csv.writer(csv_saver, delimiter=',')
        for i in range(len(pathNodesList)):
            line = pathNodesList[i]
            if print_pathnodes:
                print('~~~~~~~~~~~~~~~~~~~~~~')
                print(line.means[0])
                print(line.inputCommands)
                print(line.parent)

            means = line.means
            inputCommands = line.inputCommands
            parent = line.parent
            if parent is None:
                parent = -1

            data_list = [parent]
            for m in means:
                data_list.extend(m.reshape([3]).tolist())
            for ic in inputCommands:
                data_list.extend(ic.tolist())
            csv_writer.writerow(data_list)
    return


def load_and_plot_tree(filename, tree=None):
    path_in = os.path.join(SAVEPATH, filename)
    pathNodesList = pickle_import(path_in)
    plot_saved_data(pathNodesList, filename, tree)
    return


def make_tree(seed=SEED, save_csv=False, print_summary=True, print_all_nodes=False):
    # Reset the global random seed
    npr.seed(seed)

    # Call a dummy problem to make the IPOPT banner show up first
    dummy_problem_ipopt()

    # Define the starting parameters
    start_param = make_start_parameters()

    # Grow DR-RRTStar tree 
    t_start = time.time()
    dr_rrtstar = Tree(start_param)
    pathNodesList = dr_rrtstar.ExpandTree()
    t_end = time.time()

    if print_summary:
        # Print diagnostic info
        print("Number of Nodes:", len(pathNodesList))
        print('Elapsed Total Time:', t_end - t_start, ' seconds')
        print('Time suffix for saved files: ', SAVETIME)
    if print_all_nodes:
        fmt = '{:<8}{:<20}{}'
        print("Final tree all nodes")
        print(fmt.format('Node_ID', 'x', 'y'))
        for k, node in enumerate(pathNodesList):
            print(fmt.format(k,
                             round(node.means[-1, 0, :][0], 2),
                             round(node.means[-1, 1, :][0], 2)))

    # Pickle the node_list data and dump it for further analysis and plotting
    filename = 'NodeListData_' + FILEVERSION + '_' + SAVETIME
    pickle_export(SAVEPATH, filename, pathNodesList)
    if save_csv:
        write_pathnodes_csv(pathNodesList, filename)
    return filename


if __name__ == '__main__':
    # Close any existing figure
    plt.close('all')

    filename = make_tree()  # Make tree from scratch
    # filename = 'NodeListData_v2_0_1627919057'  # Load premade tree

    load_and_plot_tree(filename)
