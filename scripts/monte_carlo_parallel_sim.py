from monte_carlo_function import monte_carlo_function
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from config import SIGMAW

def collect_result(res):
    print("@@@@@@@@@@@ DONE with simulation using SigmaW = ", res, "@@@@@@@@@@@")
    return

def parallel_sim(save_time_prefix, version_number, noise_dist, num_trials, trials_offset, controller_str_list,
                 sigmaw_list, run_flag, iros_data, short_traj, plot_figs):

    num_cpus_to_use = mp.cpu_count() - 1
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Using ', num_cpus_to_use, ' CPU threads')
    pool = mp.Pool(num_cpus_to_use)
    for sigmaw in sigmaw_list:
        res = pool.apply_async(monte_carlo_function,
                         args=(
                         save_time_prefix, version_number, noise_dist, num_trials, trials_offset, controller_str_list,
                         sigmaw, run_flag, iros_data, short_traj, plot_figs),
                         callback=collect_result)
    pool.close()
    pool.join()
    print('')
    return


if __name__ == "__main__":
    save_time_prefix = "1627413080"
    version_number = "v2_0"
    plt.close('all')
    noise_dist = 'lap'  # "nrm", "lap", "gum"
    num_trials = 1000  # number of runs to perform
    trials_offset = 0  # indices to skip when saving the runs
    run_flag = True  # Set this true to run new Monte Carlo trials, set to false to pull in saved data
    controller_str_list = ['open-loop', 'lqr', 'lqrm', 'nmpc']  # controllers to use
    # controller_str_list = ['open-loop']  # controllers to use
    iros_data = True
    short_traj = True
    plot_figs = False

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
    parallel_sim(save_time_prefix, version_number, noise_dist, num_trials, trials_offset, controller_str_list,
                 sigmaw_list, run_flag, iros_data, short_traj, plot_figs)