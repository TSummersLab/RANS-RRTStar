from monte_carlo import *
import matplotlib.pyplot as plt
from tracking_controller import dtime_dynamics, OpenLoopController, LQRController
from collision_check import PtObsColFlag, LineObsColFlag
from drrrts_nmpc import drrrtstar_with_nmpc
import os

from utility.path_utility import create_directory
from utility.pickle_io import pickle_import, pickle_export
from opt_path import load_pickle_file
from utility.matrixmath import mdot

from config import SIGMAW

def monte_carlo_function(save_time_prefix, version_number, noise_dist, num_trials, trials_offset, controller_str_list, sigmaw=SIGMAW, run_flag=False, iros_data=False, short_traj=True, plot_figs=True):
    opt_traj_name = "OptTraj_"
    inputs_name = "_inputs"
    if short_traj:
        input_file = "OptTraj_short_" + version_number + "_" + save_time_prefix + "_inputs"
    else:
        input_file = "OptTraj_" + version_number + "_" + save_time_prefix + "_inputs"
    # result example: input_file = 'OptTraj_short_v2_0_1627413080_inputs'
    UNIQUE_EXP_NUM = input_file.replace(opt_traj_name, "")
    UNIQUE_EXP_NUM = UNIQUE_EXP_NUM.replace(inputs_name, "")
    # result example: UNIQUE_EXP_NUM = 'short_v2_0_1627413080'
    UNIQUE_EXP_NUM = UNIQUE_EXP_NUM + "_sigmaw_" + str(sigmaw[0,0]) + "_" + noise_dist

    if iros_data:
        MC_FOLDER = os.path.join('..', 'monte_carlo/IROS2021', UNIQUE_EXP_NUM)
    else:
        MC_FOLDER = os.path.join('..', 'monte_carlo', UNIQUE_EXP_NUM)

    PROBLEM_DATA_STR = 'problem_data'
    RESULT_DATA_STR = 'result_data'
    PICKLE_EXTENSION = '.pkl'

    x_ref_hist, u_ref_hist = load_ref_traj(input_file)

    controller_objects_and_init_time = [make_controller(controller_str, x_ref_hist, u_ref_hist) for controller_str in
                                        controller_str_list]
    controller_list = [result[0] for result in controller_objects_and_init_time]  # extract controller list
    setup_time_list = [result[1] for result in
                       controller_objects_and_init_time]  # extract time to create controller object
    # controller_list, time_list = [make_controller(controller_str, x_ref_hist, u_ref_hist) for controller_str in controller_str_list]
    common_data = {'x_ref_hist': x_ref_hist,
                   'u_ref_hist': u_ref_hist}

    if run_flag:
        # Make new problem data
        idx_list = make_problem_data(common_data, num_trials=num_trials, offset=trials_offset, dist=noise_dist, sigmaw=sigmaw, mc_folder=MC_FOLDER)
        # Run the monte carlo simulation
        monte_carlo(idx_list, common_data, controller_list, setup_time_list, verbose=True, mc_folder=MC_FOLDER)
    else:
        idx_list = make_idx_list(num_trials, offset=trials_offset)

    # # Plot all controllers together
    # result_data_dict = aggregate_results(idx_list, controller_str_list)
    # plotter(result_data_dict, common_data)
    # # Metrics
    # metric_dict = metric_controllers(result_data_dict, common_data)

    # Plot each controller separately
    for controller_str in controller_str_list:
        my_list = [controller_str]
        result_data_dict = aggregate_results(idx_list, my_list, MC_FOLDER)

        if iros_data:
            dirname_out = os.path.join('..', 'monte_carlo', 'IROS2021', 'path_plots', UNIQUE_EXP_NUM)
        else:
            dirname_out = os.path.join('..', 'monte_carlo', 'path_plots', UNIQUE_EXP_NUM)

        if plot_figs:
            fig, ax = plotter(result_data_dict, common_data)
            filename_out = 'path_plot_' + controller_str + '.png'
            create_directory(dirname_out)
            path_out = os.path.join(dirname_out, filename_out)
            fig.savefig(path_out, dpi=600)

        # Metrics
        metric_dict = metric_controllers(result_data_dict, common_data)

        print('************************************************')
        print("SigmaW[0,0] = ", sigmaw[0,0])
        print("Results saved in: ", dirname_out)
        for controller_str, c_metric_dict in metric_dict.items():
            collisions = c_metric_dict['collisions']
            print('%s failed %d / %d ' % (controller_str, int(np.sum(collisions)), num_trials))
        for controller_str, c_metric_dict in metric_dict.items():
            avg_score = c_metric_dict['score_avg']
            print('%s score average is %f ' % (controller_str, avg_score))
        for controller_str, c_metric_dict in metric_dict.items():
            run_time_avg = c_metric_dict['run_time_avg']
            print('%s average run time is %f ' % (controller_str, run_time_avg))
        for controller_str, c_metric_dict in metric_dict.items():
            if controller_str == 'nmpc':
                nlp_fail = c_metric_dict['nlp_fail']
                print('%s nlp failed %d / %d ' % (controller_str, int(np.sum(nlp_fail)), num_trials))

    print('************************************************')

    return sigmaw

if __name__ == "__main__":
    save_time_prefix = "1627413080"
    version_number = "v2_0"
    plt.close('all')
    noise_dist = 'lap'  # "nrm", "lap", "gum"
    num_trials = 1  # number of runs to perform
    trials_offset = 0  # indices to skip when saving the runs
    run_flag = False  # Set this true to run new Monte Carlo trials, set to false to pull in saved data
    # controller_str_list = ['open-loop', 'lqr', 'lqrm', 'nmpc']  # controllers to use
    controller_str_list = ['open-loop']  # controllers to use
    sigmaw = 10*SIGMAW
    iros_data = True
    short_traj = True
    plot_figs = True

    monte_carlo_function(save_time_prefix, version_number, noise_dist, num_trials, trials_offset, controller_str_list, sigmaw, run_flag, iros_data, short_traj, plot_figs)
