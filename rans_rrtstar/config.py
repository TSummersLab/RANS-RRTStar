#!/usr/bin/env python3
"""
Changelog:
New in version 1_0:
- Create a configuration file for RRT*. Functions that use RRT* outputs will use some of these configurations

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
Github:
@The-SS
"""

import numpy as np
import os

# Disable divide-by-zero warnings from NumPy
# np.seterr(divide='ignore')

# Random seed
SEED = 1

# Absolute path to this config file
CONFIGPATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# Saving Data Parameters
SAVEDATA = True  # True --> save data, False --> don't save data

# SAVEPATH = os.path.join(CONFIGPATH, 'saved_data')  # path to save data
# SAVEPATH = os.path.join(CONFIGPATH, 'saved_data', 'IROS2021')  # path to save data
SAVEPATH = os.path.join(CONFIGPATH, 'saved_data', 'Example')  # path to save data

# Path to the style sheet for matplotlib
STYLEPATH = os.path.join(CONFIGPATH, '../utility', 'conlab.mplstyle')

# RRT Parameters
NUMSAMPLES = 20  # 1200  # total number of samples
STEER_TIME = 30  # Maximum Steering Time Horizon
ENVCONSTANT = 1.1  # Environment Constant for computing search radius
DT = 0.2  # timestep between controls
RRT = False  # True --> RRT, False --> RRT*
DRRRT = True  # True --> apply DR checks, False --> regular RRT
MAXDECENTREWIRE = np.inf
RANDNODES = True  # false --> only 5 handpicked nodes for debugging
SATLIM = 1  # saturation limit (random nodes sampled will be cropped down to meet this limit from the nearest node)
SBSP = 100  # Shrinking Ball Sampling Percentage (% nodes in ball to try to rewire) (100 --> all nodes rewired)
SBSPAT = 3  # SBSP Activation Threshold (min number of nodes needed to be in the shrinking ball for this to activate)

# Robot Parameters
ROBRAD = 0.51 / 2  # radius of robot (added as padding to environment bounds and the obstacles)
# VELMIN, VELMAX = -2000, 2000
VELMIN, VELMAX = -0.5, 0.5  # min and max linear velocity limits
ANGVELMIN, ANGVELMAX = -np.pi, np.pi  # min and max angular velocity limits

# Environment Parameters
ENVNUM = 9
if ENVNUM == 0:  # Opt ctrl course project environment
    ENVAREA = [-5, 5, -5, 5]  # [xmin,xmax,ymin,ymax]
    GOALAREA = [0, 1, -1, 1]  # [xmin,xmax,ymin,ymax] Goal zone
    ROBSTART = [-3., -4.]  # robot starting location (x,y)
    RANDAREA = [-4.9, 4.9, -4.9, 4.9] # area sampled: [xmin,xmax,ymin,ymax], [-4.7, 4.7, -4.7, 4.7] good with 0 ROBRAD, limit:[-5,5,-5,5]
    OBSTACLELIST = [[-4, 0, 3, 1],
                    [-2, -2, 1, 2],
                    [-1, -2, 3, 0.5],
                    [2, -2, 1.5, 5],
                    [-4, 2, 4.5, 1]] # [ox,oy,wd,ht]
elif ENVNUM == 1:  # tea cup
    ENVAREA = [-5, 5, -5, 5]  # [xmin,xmax,ymin,ymax]
    GOALAREA = [0, 1, -1, 1]  # [xmin,xmax,ymin,ymax] Goal zone
    ROBSTART = [-3., -4.]  # robot starting location (x,y)
    RANDAREA = [-4.9, 4.9, -4.9, 4.9] # area sampled: [xmin,xmax,ymin,ymax], [-4.7, 4.7, -4.7, 4.7] good with 0 ROBRAD, limit:[-5,5,-5,5]
    OBSTACLELIST = [[-3., 0, 2.5, 0.5],
                    [-1.5, -2, 0.5, 2],
                    [-1+3*ROBRAD, -2, 2, 0.5],
                    [2, -2, 0.5, 4.5],
                    [-4, 2.0, 3.5, 0.75]]
elif ENVNUM == 2:  # fly trap
    ENVAREA = [-5, 5, -5, 5]  # [xmin,xmax,ymin,ymax]
    GOALAREA = [-1.0, 1.0, -2.5, -1.5]  # [xmin,xmax,ymin,ymax] Goal zone
    ROBSTART = [4.0, 4.0]  # robot starting location (x,y)
    RANDAREA = [-4.9, 4.9, -4.9, 4.9]
    OBSTACLELIST = [[1.5, 1, 0.5, 3.2],
                    [-3, 1, 5, 0.5],
                    [-3, -1, 5, 0.5],
                    [1.5, -3.5, 0.5, 3.],
                    [-4.3, -3.5, 6.3, 0.5]]
elif ENVNUM == 3:  # fly trap with gap
    ENVAREA = [-5, 5, -5, 5]  # [xmin,xmax,ymin,ymax]
    GOALAREA = [-1.0, 1.0, -2.5, -1.5]  # [xmin,xmax,ymin,ymax] Goal zone
    ROBSTART = [4.0, 4.0]  # robot starting location (x,y)
    RANDAREA = [-4.9, 4.9, -4.9, 4.9]
    OBSTACLELIST = [[1.5, 1.0, 0.5, 3.2],
                    [-3.0, 1.0, 5.0, 0.5],
                    [-3.0, -1.0, 5.0, 0.5],
                    [1.5, -3.5, 0.5, 1.5],
                    [-4.3, -3.5, 6.3, 0.5]]
elif ENVNUM == 4:  # three slabs maze
    ENVAREA = [-5, 5, -5, 5]  # [xmin,xmax,ymin,ymax]
    GOALAREA = [2.0, 3.5, -4.5, -3.5]  # [xmin,xmax,ymin,ymax] Goal zone
    ROBSTART = [4.0, 4.0]  # robot starting location (x,y)
    RANDAREA = [-4.9, 4.9, -4.9, 4.9]
    OBSTACLELIST = [[-2.0, 2.0, 6.0, 0.5],
                    [-4.5, -0.5, 7.0, 0.5],
                    [-3., -3., 7.0, 0.5]]
elif ENVNUM == 9:  # easy environment
    ENVAREA = [-5, 5, -5, 5]  # [xmin,xmax,ymin,ymax]
    GOALAREA = [1.0, 4.0, -2.0, 2.0]  # [xmin,xmax,ymin,ymax] Goal zone
    ROBSTART = [3.0, 3.0]  # robot starting location (x,y)
    RANDAREA = [-4.9, 4.9, -4.9, 4.9]
    OBSTACLELIST = [[-1.0, -1.0, 1.0, 1.0]]
else:
    raise ValueError('Invalid environment key!')


# Noise parameters
SIGMAW = np.diag([0.0000005, 0.0000005, 0.0000005])  # Covariance of process noise
# SIGMAW = np.diag([0.001, 0.001, 0.001])  # Covariance of process noise
# print('3 sigma value: ', 3*SIGMAW[0, 0]**0.5)
SIGMAV = np.diag([0., 0., 0.])  # Covariance of sensor noise (we don't have any for now)
CROSSCOR = np.diag([0., 0., 0.])  # Cross Correlation between the two noises (none for now)

# DR Risk bounds
BETA = 0.1  # desired risk bound for entire plan failure
TMAX = 1000  # maximum number of trajectory points in the RRT* plan (not just RRT* node, but also the intermediate trajectory points)
num_obs = len(OBSTACLELIST)
num_constraints = 4*(num_obs + 1) # total number of constraints for rectangular environment and rectangular obstacles
alfa_stage = BETA/TMAX  # stage risk bound
alfa_const = alfa_stage/num_constraints  # constraint risk bound for that stage
ALFA = [alfa_const] * (len(OBSTACLELIST)+4)  # risk bound for each obstacle + each environment side will be treated as an obstacle, so we add their risk bound at the end 4 times

# High Level Planner quadratic cost function parameters
QHL = np.diag([1.0, 1.0, 0.001]) # state quadratic cost matrix for high level plan
RHL = 100 * np.diag([1.0, 1.0]) # cost quadratic cost matrix for high level plan

# Low Level Tracker quadratic cost function parameters
QLL = 100 * np.diag([1.0, 1.0, 0.1])
RLL = 1 * np.diag([1.0, 1.0])
QTLL = 10 * QLL
