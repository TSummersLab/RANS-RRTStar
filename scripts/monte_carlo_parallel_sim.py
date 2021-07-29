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
    num_trials = 10  # number of runs to perform
    trials_offset = 0  # indices to skip when saving the runs
    run_flag = False  # Set this true to run new Monte Carlo trials, set to false to pull in saved data
    # controller_str_list = ['open-loop', 'lqr', 'lqrm', 'nmpc']  # controllers to use
    controller_str_list = ['open-loop']  # controllers to use
    iros_data = True
    short_traj = True
    plot_figs = True

    sigmaw_list = []
    sigmaw_list.append(np.diag([0.0000005, 0.0000005, 0.0000005]))
    sigmaw_list.append(np.diag([0.000001,  0.000001,  0.000001]))
    sigmaw_list.append(np.diag([0.000005,  0.000005,  0.000005]))
    sigmaw_list.append(np.diag([0.00001,   0.00001,   0.00001]))
    sigmaw_list.append(np.diag([0.00005,   0.00005,   0.00005]))
    sigmaw_list.append(np.diag([0.0001,    0.0001,    0.0001]))
    sigmaw_list.append(np.diag([0.0005,    0.0005,    0.0005]))
    sigmaw_list.append(np.diag([0.001,     0.001,     0.001]))
    sigmaw_list.append(np.diag([0.005,     0.005,     0.005]))
    sigmaw_list.append(np.diag([0.01,      0.01,      0.01]))
    sigmaw_list.append(np.diag([0.05,      0.05,      0.05]))
    sigmaw_list.append(np.diag([0.1,       0.1,       0.1]))
    sigmaw_list.append(np.diag([0.5,       0.5,       0.5]))
    parallel_sim(save_time_prefix, version_number, noise_dist, num_trials, trials_offset, controller_str_list,
                 sigmaw_list, run_flag, iros_data, short_traj, plot_figs)