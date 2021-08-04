#!/usr/bin/env python3
"""
Changelog:
New is v1_0:
- Create script

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author:
Benjamin Gravell
Email:
benjamin.gravell@utdallas.edu
Github:
@BenGravell

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
Github:
@The-SS

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script runs Monte Carlo trials of the nonlinear system for a given environment and reference trajectory
at various noise levels.

"""

import os
import time
import multiprocessing as mp

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

from rans_rrtstar.config import DT, ENVAREA, RANDAREA, VELMIN, VELMAX, ANGVELMIN, ANGVELMAX,\
                   OBSTACLELIST, ROBRAD, QLL, RLL, QTLL, SIGMAW, SAVEPATH
from rans_rrtstar.common.dynamics import DYN
from rans_rrtstar.common.tracking_controller import OpenLoopController, LQRController, create_lqrm_controller
from rans_rrtstar.common.collision_check import point_obstacle_collision_flag, line_obstacle_collision_flag
from rans_rrtstar.common.drrrts_nmpc import drrrtstar_with_nmpc
from rans_rrtstar.common.plotting import plot_paths
from rans_rrtstar.filesearch import get_timestr

from utility.path_utility import create_directory
from utility.pickle_io import pickle_import, pickle_export
from utility.matrixmath import mdot


PROBLEM_DATA_STR = 'problem_data'
RESULT_DATA_STR = 'result_data'
PICKLE_EXTENSION = '.pkl'


def load_ref_traj(input_file):
    inputs_string = "inputs"
    states_file = input_file.replace(inputs_string, "states")

    input_file = os.path.join(SAVEPATH, input_file)
    states_file = os.path.join(SAVEPATH, states_file)

    # load inputs and states
    ref_inputs = pickle_import(input_file)
    ref_states = pickle_import(states_file)

    failures = DYN.check_entire_traj(ref_states, ref_inputs)

    return ref_states, ref_inputs


def idx2str(idx):
    # This function establishes a common data directory naming specification
    return '%012d' % idx


def import_problem_data(idx, mc_folder):
    path_in = os.path.join(mc_folder, idx2str(idx), PROBLEM_DATA_STR+PICKLE_EXTENSION)
    problem_data = pickle_import(path_in)
    return problem_data


def export_problem_data(problem_data, idx, mc_folder):
    dirname_out = os.path.join(mc_folder, idx2str(idx))
    filename_out = PROBLEM_DATA_STR+PICKLE_EXTENSION
    pickle_export(dirname_out, filename_out, problem_data)
    return


def import_result_data(idx, controller_str, mc_folder):
    path_in = os.path.join(mc_folder, idx2str(idx), RESULT_DATA_STR+'_'+controller_str+PICKLE_EXTENSION)
    result = pickle_import(path_in)
    return result


def export_result_data(problem_data, idx, controller_str, mc_folder):
    dirname_out = os.path.join(mc_folder, idx2str(idx))
    filename_out = RESULT_DATA_STR+'_'+controller_str+PICKLE_EXTENSION
    pickle_export(dirname_out, filename_out, problem_data)
    return


def generate_disturbance_history(common_data, seed=None, dist=None, show_hist=False, sigmaw=SIGMAW):
    rng = npr.default_rng(seed)
    sigma1 = sigmaw[0, 0]  # first entry in SigmaW
    x_ref_hist = common_data['x_ref_hist']
    T = x_ref_hist.shape[0]

    if dist is None:
        dist = "nrm"  # "nrm", "lap", "gum"

    if dist == "nrm":
        w_hist = rng.multivariate_normal(mean=[0, 0, 0], cov=sigmaw, size=T)
    elif dist == "lap":
        l = 0
        b = (sigma1 / 2) ** 0.5
        w_hist = rng.laplace(loc=l, scale=b, size=[T, 3])  # mean = loc, var = 2*scale^2
    elif dist == "gum":
        b = (6*sigma1)**0.5/np.pi
        l = -0.57721*b
        w_hist = rng.gumbel(loc=l, scale=b, size=[T, 3])  # mean = loc+0.57721*scale, var = pi^2/6 scale^2
    elif dist == "none":
        w_hist = 0*rng.multivariate_normal(mean=[0, 0, 0], cov=sigmaw, size=T)
    else:
        raise ValueError('Invalid disturbance generation method!')

    if show_hist:
        plt.hist(w_hist)
        plt.show()
    return w_hist


def make_idx_list(num_trials, offset=0):
    idx_list = []
    for i in range(num_trials):
        idx = i+offset+1
        idx_list.append(idx)
    return idx_list


def make_problem_data(T, num_trials, offset=0, dist='nrm', sigmaw=SIGMAW, mc_folder=None):
    idx_list = []
    for i in range(num_trials):
        idx = i + offset + 1
        w_hist = generate_disturbance_history(T, seed=idx, dist=dist, sigmaw=sigmaw)
        problem_data = {'w_hist': w_hist}
        export_problem_data(problem_data, idx, mc_folder)
        idx_list.append(idx)
    return idx_list


class nmpc_controller_object():  # place holder for nmpc controller object
    def __init__(self):
        self.name = 'nmpc'


def make_controller(controller_str, x_ref_hist, u_ref_hist):
    """
    Creates the controller object and returns the time it took to do so
    """

    time_start = time.time()
    if controller_str == 'open-loop':
        # Create open-loop controller object
        controller = OpenLoopController(u_ref_hist)
    elif controller_str == 'lqr':
        # Create vanilla LQR controller object
        controller = create_lqrm_controller(x_ref_hist, u_ref_hist, use_robust_lqr=False)
    elif controller_str == 'lqrm':
        # Create robust LQR controller object
        controller = create_lqrm_controller(x_ref_hist, u_ref_hist, use_robust_lqr=True)
    elif controller_str == 'nmpc':
        # Create nmpc controller object
        controller = nmpc_controller_object()
    else:
        raise ValueError('Invalid controller string')
    controller.name = controller_str
    time_stop = time.time()
    return [controller, time_stop - time_start]


def rollout(n, m, T, DT, x0=None, w_hist=None, controller=None, saturate_inputs=True):
    # Initialize
    x_hist = np.zeros([T, n])
    if x0 is None:
        x0 = np.zeros(n)
    x_hist[0] = x0
    x = np.copy(x0)
    u_hist = np.zeros([T, m])

    collision_flag = False
    collision_idx = None

    # Simulate
    for t in range(T-1):
        # Compute desired control inputs
        u = controller.compute_input(x, t)
        # Saturate inputs at actuator limits
        if saturate_inputs:
            u[0] = np.clip(u[0], VELMIN, VELMAX)
            u[1] = np.clip(u[1], ANGVELMIN, ANGVELMAX)
        # Get disturbance
        w = w_hist[t]
        # Transition the state
        x_old = np.copy(x)
        x = DYN.dtime_dynamics(x, u) + w
        # Check for collision
        if point_obstacle_collision_flag(x, OBSTACLELIST, RANDAREA, ROBRAD) or line_obstacle_collision_flag(x_old, x, OBSTACLELIST, ROBRAD):
            collision_flag = True
            collision_idx = t
            x_hist[t+1:] = x  # pad out x_hist with the post-collision state
            break
        # Record quantities
        x_hist[t+1] = x
        u_hist[t] = u

    result_data = {'x_hist': x_hist,
                   'u_hist': u_hist,
                   'collision_flag': collision_flag,
                   'collision_idx': collision_idx}
    return result_data


def trial(problem_data, common_data, controller, setup_time):
    # Get the reference trajectory and controller
    x_ref_hist = common_data['x_ref_hist']
    u_ref_hist = common_data['u_ref_hist']

    # Number of states, inputs
    T, n = x_ref_hist.shape
    T, m = u_ref_hist.shape

    # Start in the reference initial state
    x0 = x_ref_hist[0]

    # Get the disturbance for this trial
    w_hist = problem_data['w_hist']

    # Simulate trajectory with noise and control, forwards in time
    time_start = time.time()
    if type(controller) in [OpenLoopController, LQRController]:
        result_data = rollout(n, m, T, DT, x0, w_hist, controller=controller)
    elif controller.name == 'nmpc':
        nmpc_horizon = 10
        result_data = drrrtstar_with_nmpc(nmpc_horizon, x_ref_hist, u_ref_hist, n, m, T, w=w_hist, drnmpc=False, hnmpc=False)
    else:
        raise NotImplementedError('Need rollout function for NMPC!')
    time_stop = time.time()
    sim_time = time_stop - time_start

    if 'run_time' in result_data.keys():
        result_data['run_time'] += setup_time
    else:
        result_data['run_time'] = setup_time + sim_time
    return result_data


def monte_carlo(idx_list, common_data, controller_list, setup_time_list, verbose=False, mc_folder=None):
    num_trials = len(idx_list)
    for i, idx in enumerate(idx_list):
        if verbose:
            print('Trial %6d / %d    ' % (i+1, num_trials), end='')
            print('Problem %s    ' % idx2str(idx), end='')
        problem_data = import_problem_data(idx, mc_folder)

        if verbose:
            print('Simulating...', end='')
        for j, controller in enumerate(controller_list):
            result_data = trial(problem_data, common_data, controller, setup_time_list[j])
            export_result_data(result_data, idx, controller.name, mc_folder)
        if verbose:
            print(' complete.')
    return


def aggregate_results(idx_list, controller_str_list, mc_folder):
    result_data_dict = {}
    for controller_str in controller_str_list:
        result_data_list = []
        for idx in idx_list:
            result_data = import_result_data(idx, controller_str, mc_folder)
            result_data_list.append(result_data)
        result_data_dict[controller_str] = result_data_list
    return result_data_dict


def cost_of_trajectory(x_ref_hist, x_hist, u_hist):
    T = x_hist.shape[0]
    dxtot = 0
    utot = 0
    for t in range(T):
        if t < T-1:
            Q = QLL
        else:
            Q = QTLL
        R = RLL
        dx = x_hist[t] - x_ref_hist[t]
        u = u_hist[t]
        dxtot += mdot(dx.T, Q, dx)
        utot += mdot(u.T, R, u)
    return dxtot, utot


def score_trajectory(x_ref_hist, u_ref_hist, x_hist, u_hist):
    # NOTE: x_hist is assumed to be collision-free
    return cost_of_trajectory(x_ref_hist, x_hist, u_hist)


def metric_trials(result_data_list, common_data, skip_scores=False):
    x_ref_hist = common_data['x_ref_hist']
    u_ref_hist = common_data['u_ref_hist']
    N = len(result_data_list)
    num_collisions = 0
    collisions = np.full(N, False)
    dx_score_sum = 0.0
    dx_scores = np.zeros(N)
    u_score_sum = 0.0
    u_scores = np.zeros(N)
    run_time_sum = 0.0
    nlp_fail = np.full(N, False)
    for i, result_data in enumerate(result_data_list):
        x_hist = result_data['x_hist']
        u_hist = result_data['u_hist']
        collision_flag = result_data['collision_flag']
        run_time = result_data['run_time']
        run_time_sum += run_time
        try:
            nlp_fail_flag = result_data["nlp_failed_flag"]
            if nlp_fail_flag:
                nlp_fail[i] = True
        except :
            pass
        if collision_flag:
            num_collisions += 1
            collisions[i] = True
            dx_scores[i] = np.inf
            u_scores[i] = np.inf
        else:
            if skip_scores:
                dx_score, u_score = -1, -1
            else:
                dx_score, u_score = score_trajectory(x_ref_hist, u_ref_hist, x_hist, u_hist)
            dx_scores[i] = dx_score
            dx_score_sum += dx_score
            u_scores[i] = u_score
            u_score_sum += u_score
    run_time_avg = run_time_sum/N
    if num_collisions < N:
        dx_score_avg = dx_score_sum/(N-num_collisions)
        u_score_avg = u_score_sum/(N-num_collisions)
    else:
        dx_score_avg = np.inf
        u_score_avg = np.inf
    collision_avg = num_collisions/N

    out_dict = {'dx_scores': dx_scores,
                'dx_score_avg': dx_score_avg,
                'u_scores': u_scores,
                'u_score_avg': u_score_avg,
                'collisions': collisions,
                'collision_avg': collision_avg,
                'run_time_avg': run_time_avg,
                'nlp_fail': nlp_fail}
    return out_dict


def metric_controllers(result_data_dict, common_data, skip_scores=False):
    metric_dict = {}
    for controller_str, result_data_list in result_data_dict.items():
        metric_dict[controller_str] = metric_trials(result_data_list, common_data, skip_scores)
    return metric_dict


def score_histogram(score_dict):
    fig, ax = plt.subplots()
    for controller_str, c_score_dict in score_dict.items():
        scores = c_score_dict['scores']
        num_bins = 40
        ax.hist(scores[np.isfinite(scores)], bins=num_bins, density=True, alpha=0.5, label=controller_str)
    ax.legend()
    return fig, ax


def plotter(result_data_dict, common_data):
    x_ref_hist = common_data['x_ref_hist']
    u_ref_hist = common_data['u_ref_hist']
    T, n = x_ref_hist.shape
    T, m = u_ref_hist.shape
    t_hist = np.arange(T) * DT

    x_hist_all_dict = {}
    collision_flag_all_dict = {}

    for controller_str, result_data_list in result_data_dict.items():
        x_hist_all = np.array([result_data['x_hist'] for result_data in result_data_list])
        collision_flag_all = [result_data['collision_flag'] for result_data in result_data_list]
        x_hist_all_dict[controller_str] = x_hist_all
        collision_flag_all_dict[controller_str] = collision_flag_all

    # Plot Monte Carlo paths
    ax_lim = np.array(ENVAREA) + np.array([-0.2, 0.2, -0.2, 0.2])
    fig, ax = plot_paths(t_hist, x_hist_all_dict, collision_flag_all_dict, x_ref_hist, title=None, fig_offset=None,  axis_limits=ax_lim)
    return fig, ax


def monte_carlo_function(timestr, noise_dist, num_trials, trials_offset, controller_str_list,
                         sigmaw, run_flag=False, short_traj=True, plot_figs=True):
    opt_traj_name = "OptTraj_"
    inputs_name = "_inputs"
    version_number = 'v2_0'
    if short_traj:
        input_file = opt_traj_name + "short_" + version_number + "_" + timestr + inputs_name
    else:
        input_file = opt_traj_name + version_number + "_" + timestr + inputs_name
    # result example: input_file = 'OptTraj_short_v2_0_1627413080_inputs'
    UNIQUE_EXP_NUM = input_file.replace(opt_traj_name, "")
    UNIQUE_EXP_NUM = UNIQUE_EXP_NUM.replace(inputs_name, "")
    # result example: UNIQUE_EXP_NUM = 'short_v2_0_1627413080'
    UNIQUE_EXP_NUM = UNIQUE_EXP_NUM + "_sigmaw_" + str(sigmaw[0, 0]) + "_" + noise_dist

    MC_FOLDER = os.path.join(SAVEPATH, 'monte_carlo', UNIQUE_EXP_NUM)

    x_ref_hist, u_ref_hist = load_ref_traj(input_file)

    controller_objects_and_init_time = [make_controller(controller_str, x_ref_hist, u_ref_hist) for controller_str in
                                        controller_str_list]
    controller_list = [result[0] for result in controller_objects_and_init_time]  # extract controller list
    setup_time_list = [result[1] for result in
                       controller_objects_and_init_time]  # extract time to create controller object
    common_data = {'x_ref_hist': x_ref_hist,
                   'u_ref_hist': u_ref_hist}

    if run_flag:
        # Make new problem data
        idx_list = make_problem_data(common_data, num_trials=num_trials, offset=trials_offset, dist=noise_dist,
                                     sigmaw=sigmaw, mc_folder=MC_FOLDER)
        # Run the monte carlo simulation
        monte_carlo(idx_list, common_data, controller_list, setup_time_list, verbose=True, mc_folder=MC_FOLDER)
    else:
        idx_list = make_idx_list(num_trials, offset=trials_offset)

    # Plotting and metrics
    plot_controllers_together = False

    if plot_controllers_together:
        result_data_dict = aggregate_results(idx_list, controller_str_list, mc_folder=MC_FOLDER)
        plotter(result_data_dict, common_data)
        metric_dict = metric_controllers(result_data_dict, common_data)
    else:
        # Plot each controller separately
        for controller_str in controller_str_list:
            my_list = [controller_str]
            result_data_dict = aggregate_results(idx_list, my_list, MC_FOLDER)

            dirname_out = os.path.join(SAVEPATH, 'monte_carlo', 'path_plots', UNIQUE_EXP_NUM)

            if plot_figs:
                fig, ax = plotter(result_data_dict, common_data)
                filename_out = 'path_plot_' + controller_str + '.png'
                create_directory(dirname_out)
                path_out = os.path.join(dirname_out, filename_out)
                fig.savefig(path_out, dpi=600)

            # Metrics
            metric_dict = metric_controllers(result_data_dict, common_data)

            print('************************************************')
            print("SigmaW[0,0] = ", sigmaw[0, 0])
            print("Results saved in: ", dirname_out)
            for controller_str, c_metric_dict in metric_dict.items():
                collisions = c_metric_dict['collisions']
                print('%s failed %d / %d '%(controller_str, int(np.sum(collisions)), num_trials))
            for controller_str, c_metric_dict in metric_dict.items():
                dx_score_avg = c_metric_dict['dx_score_avg']
                print('%s dx_score average is %f '%(controller_str, dx_score_avg))
            for controller_str, c_metric_dict in metric_dict.items():
                u_score_avg = c_metric_dict['u_score_avg']
                print('%s u_score average is %f '%(controller_str, u_score_avg))
            for controller_str, c_metric_dict in metric_dict.items():
                run_time_avg = c_metric_dict['run_time_avg']
                print('%s average run time is %f '%(controller_str, run_time_avg))
            for controller_str, c_metric_dict in metric_dict.items():
                if controller_str == 'nmpc':
                    nlp_fail = c_metric_dict['nlp_fail']
                    print('%s nlp failed %d / %d '%(controller_str, int(np.sum(nlp_fail)), num_trials))

        print('************************************************')

    return sigmaw


def collect_result(res):
    print("@@@@@@@@@@@ DONE with simulation using SigmaW = ", res, "@@@@@@@@@@@")
    return


def multisim(timestr, noise_dist, num_trials, trials_offset, controller_str_list,
                 sigmaw_list, run_flag, short_traj, plot_figs, parallel_option='parallel'):
    if parallel_option == 'parallel':
        num_cpus_to_use = mp.cpu_count() - 1  # Leave one cpu open so computer does not lock up
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Using ', num_cpus_to_use, ' CPU threads')
        pool = mp.Pool(num_cpus_to_use)
        for sigmaw in sigmaw_list:
            res = pool.apply_async(monte_carlo_function,
                             args=(
                                 timestr, noise_dist, num_trials, trials_offset, controller_str_list,
                                 sigmaw, run_flag, short_traj, plot_figs),
                             callback=collect_result)
        pool.close()
        pool.join()
        print('')
    elif parallel_option == 'serial':
        for sigmaw in sigmaw_list:
            res = monte_carlo_function(timestr, noise_dist, num_trials, trials_offset,
                                       controller_str_list,
                                       sigmaw, run_flag, short_traj, plot_figs)
            collect_result(res)
    return


def make_sigmaw_list():
    sigmaw_list = []

    sigmaw_list.append(0.0000005 * np.eye(3))
    sigmaw_list.append(0.000001 * np.eye(3))
    sigmaw_list.append(0.000005 * np.eye(3))
    sigmaw_list.append(0.00001 * np.eye(3))
    sigmaw_list.append(0.00005 * np.eye(3))
    sigmaw_list.append(0.0001 * np.eye(3))
    sigmaw_list.append(0.00015 * np.eye(3))
    sigmaw_list.append(0.0002 * np.eye(3))
    sigmaw_list.append(0.00025 * np.eye(3))
    sigmaw_list.append(0.0003 * np.eye(3))
    sigmaw_list.append(0.00035 * np.eye(3))
    sigmaw_list.append(0.0004 * np.eye(3))
    sigmaw_list.append(0.00045 * np.eye(3))
    sigmaw_list.append(0.0005 * np.eye(3))
    sigmaw_list.append(0.00055 * np.eye(3))
    sigmaw_list.append(0.0006 * np.eye(3))
    sigmaw_list.append(0.00065 * np.eye(3))
    sigmaw_list.append(0.0007 * np.eye(3))
    sigmaw_list.append(0.00075 * np.eye(3))
    sigmaw_list.append(0.0008 * np.eye(3))
    sigmaw_list.append(0.00085 * np.eye(3))
    sigmaw_list.append(0.0009 * np.eye(3))
    sigmaw_list.append(0.00095 * np.eye(3))
    sigmaw_list.append(0.001 * np.eye(3))
    sigmaw_list.append(0.0015 * np.eye(3))
    sigmaw_list.append(0.002 * np.eye(3))
    sigmaw_list.append(0.0025 * np.eye(3))
    sigmaw_list.append(0.003 * np.eye(3))
    sigmaw_list.append(0.0035 * np.eye(3))
    sigmaw_list.append(0.004 * np.eye(3))
    sigmaw_list.append(0.0045 * np.eye(3))
    sigmaw_list.append(0.005 * np.eye(3))
    sigmaw_list.append(0.0055 * np.eye(3))
    sigmaw_list.append(0.006 * np.eye(3))
    sigmaw_list.append(0.0065 * np.eye(3))
    sigmaw_list.append(0.007 * np.eye(3))
    sigmaw_list.append(0.0075 * np.eye(3))
    sigmaw_list.append(0.008 * np.eye(3))
    sigmaw_list.append(0.0085 * np.eye(3))
    sigmaw_list.append(0.009 * np.eye(3))
    sigmaw_list.append(0.0095 * np.eye(3))
    sigmaw_list.append(0.01 * np.eye(3))
    # sigmaw_list.append(0.015 * np.eye(3))
    sigmaw_list.append(0.02 * np.eye(3))
    # sigmaw_list.append(0.025 * np.eye(3))
    sigmaw_list.append(0.03 * np.eye(3))
    # sigmaw_list.append(0.035 * np.eye(3))
    sigmaw_list.append(0.04 * np.eye(3))
    # sigmaw_list.append(0.045 * np.eye(3))
    sigmaw_list.append(0.05 * np.eye(3))
    # sigmaw_list.append(0.055 * np.eye(3))
    sigmaw_list.append(0.06 * np.eye(3))
    # sigmaw_list.append(0.065 * np.eye(3))
    sigmaw_list.append(0.07 * np.eye(3))
    # sigmaw_list.append(0.075 * np.eye(3))
    sigmaw_list.append(0.08 * np.eye(3))
    # sigmaw_list.append(0.085 * np.eye(3))
    sigmaw_list.append(0.09 * np.eye(3))
    # sigmaw_list.append(0.095 * np.eye(3))
    sigmaw_list.append(0.1 * np.eye(3))

    return sigmaw_list


def monte_carlo_main(timestr, draft=False):
    if draft:
        # Run Monte Carlo trials with draft settings to get quick outputs
        noise_dist = 'lap'  # "nrm", "lap", "gum"
        num_trials = 10  # number of runs to perform
        trials_offset = 0  # indices to skip when saving the runs
        run_flag = True  # Set this true to run new Monte Carlo trials, set to false to pull in saved data

        controller_str_list = ['open-loop', 'lqr', 'lqrm']  # controllers to use

        sigmaw_list = [s*np.eye(3) for s in [0.0001, 0.001]]

        short_traj = True
        plot_figs = True
    else:
        # Run Monte Carlo trials with final settings to get high quality outputs
        noise_dist = 'lap'  # "nrm", "lap", "gum"
        num_trials = 1000  # number of runs to perform
        trials_offset = 0  # indices to skip when saving the runs
        run_flag = True  # Set this true to run new Monte Carlo trials, set to false to pull in saved data

        controller_str_list = ['open-loop', 'lqr', 'lqrm', 'nmpc']  # controllers to use

        sigmaw_list = make_sigmaw_list()

        short_traj = True
        plot_figs = False

    multisim(timestr, noise_dist, num_trials, trials_offset, controller_str_list,
             sigmaw_list, run_flag, short_traj, plot_figs, parallel_option='serial')


if __name__ == "__main__":
    timestr = get_timestr(SAVEPATH)

    plt.close('all')

    monte_carlo_main(timestr, draft=True)
    # monte_carlo_main(timestr, draft=False)
