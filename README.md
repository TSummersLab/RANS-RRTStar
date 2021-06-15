# RANS-RRT*
This is an implementation of our work 
["Risk-Averse RRT* Planning with Nonlinear Steering and Tracking
Controllers for Nonlinear Robotic Systems Under Uncertainty"](https://arxiv.org/pdf/2103.05572.pdf).

RANS-RRT* (Risk-Averse Nonlinear Steering RRT*) is a sampling-based planning algorithm based on RRT*. 
The current implementation uses unicycle dynamics for the nonlinear steering function. 
The generated trajectory can be passed down to low-level tracking controllers: 
LQR, 
LQR with multiplicative noise terms,
and nonlinear model predictive control (NMPC).

**Authors:**
- [Sleiman Safaoui - UT Dallas](https://github.com/The-SS)
- [Ben Gravell - UT Dallas](https://github.com/BenGravell)
- [Venkatraman Renganathan - UT Dallas](https://github.com/venkatramanrenganathan)
- [Dr. Tyler Summers - UT Dallas](https://github.com/tsummers)

# Table of Contents
- [Cloning the Package](#cloning-the-package)
- [Dependencies](#dependencies)
- [Package Organization and Description](#package-organization-and-description)
- [Using the Package](#using-the-package)
- [License](#license)

## Cloning the Package
The package contains submodules. Please clone with:
```
git clone https://github.com/TSummersLab/RANS-RRTStar.git --recurse
```
Note: When using GitHub desktop, the application initializes all submodules when cloning for the first time.
If a submodule is not initialized after pulling changes, please use the Git Bash tool or terminal and run 
`git submodule update --init` at the root of the repository.

## Dependencies
The package uses Python 3 and requires the following packages:
- NumPy
- SciPy
- CasADi
- namedlist

## Package Organization and Description
The main scripts are organized as follows:
```
config.py
rans_rrtstar.py
get_opt_paths.py
scripts
    |__monte_carlo.py
saved_data
monte_carlo
```
### Config File
`config.py` provides a list of stetting that can be changed by the user (all are documented in the file). 
We note the following parameters:
- `NUMSAMPLES`: Number of random samples that RRT* will try to add. Not all will be added to the tree.
- `STEER_TIME`: Number of discrete time steps used for steering between two nodes.
- `RRT`: Switches between RRT (not asymptotically optimal) and RRT* (asymptotically optimal) implementations
- `DRRRT`: Switches between using a distributionally robust collision check and not using it.

| `RRT` | `DRRT` | Resulting Algorithm |
| ----------- | ----------- | -----------|
| True | False | RRT (with nonlinear dynamics) | 
| True | True | DR-RRT (with nonlinear dynamics) |
| False | False | RRT* (with nonlinear dynamics) |
| False | True | RANS-RRT* |
- `ENVNUM`: Index for the environment. 
  We provide five environments. More can be added following the same format.
-`SAVEPATH`: Directory for saving data relative to the root of the package. 
  We recommend storing the results in new directories in `saved_data`.
- `SIGMAW`: Process noise covariance. Must be balanced with `BETA` (see below).
- `SIGMAV`: Measurement noise covariance. We do NOT use this. Keep all zeros.
- `CROSSCOR`: Cross correlation between process and measurement noise covariance. We do NOT use this. Keep all zeros.
- `BETA`: Risk bound (% of experiments that can fail). This is an upper bound. 
  The smaller it is, the larger the tightening.
- `TMAX`: Maximum number of time steps used for planning.

### RANS-RRT* File
`rans_rrtstar.py` contains all the code for RANS-RRT* 
(data structures, functions and methods for the planner, and functions for storing the resulting data).
After this script runs, two files are stored in the directory specified in the config file: 
    * the node list data (RANS-RRT* tree): `NodeListData_v<version>_<timestamp>` (pickle file)
    * an image of the generated tree: `plot_tree_v<version>_<timestamp>` (png)
### Post-Processing Trajectory File
`get_opt_paths.py` takes tree generated by `rans_rrtstar.py` and extracts the optimal trajectory from the root to the goal.
It saves the inputs and states corresponding to that trajectory, and a png image of it, in the save directory:
* `OptTraj_v*_inputs` (optimal trajectory inputs)
* `OptTraj_v*_states` (optimal trajectory states)
* `OptTraj_v*_plot.png` (optimal trajectory image)

Furthermore, it shortens the trajectory by replanning between successive RANS-RRT* nodes using a dynamic number of discrete time steps (instead of using `STEER_TIME` steps).
The shortened trajectory is also checked to satisfy the distributionally robust collision check. 
The shortened trajectory inputs, states, and image are also saved in the same directory as:
* `OptTraj_short_v*_inputs` (shortened optimal trajectory inputs)
* `OptTraj_short_v*_states` (shortened optimal trajectory states)
* `OptTraj_short_v*_plot.png` (shortened optimal trajectory image)

### Monte Carlo File
`scripts/monte_carlo` takes the post-processed data from `get_opt_paths.py` and performs tracking simulations with realized noise.
The Monte Carlo results are stored in the directory specified by the `MC_FOLDER` variable. 
Each experiment setup (new environment, new noise distribution, ...), should have a unique `UNIQUE_EXP_NUM`. 
Preferably, use a six digit number (you can also use a descriptive name). 
For a that experiment setup, each experiment run (with its own noise realizations) will be stored in a folder named by a 12-digit number (e.g.`000000000001`).
These folders are automatically generated based on the `num_trials` and `trials_offset` (see later).

In `main` of the script, the following variables can be modified.
* The trajectory specified by the chosen input file `input_file` (e.g. `OptTraj_short_v2_0_1623778273_inputs`).
* The noise distribution is specified using the `noise_dist` variable.
The available distributions are:
    * `nrm`: Gaussian/normal distribution
    * `lap`: Laplacian distribution
    * `gum`: Gumbel distribution
* The desired low-level controllers specified in the `controller_str_list` list.
The available controllers are: 
    * `'open-loop'`: open loop control
    * `'lqr'`: LQR control
    * `'lqrm'`: LQR with multiplicative terms control
    * `'nmpc'`: nonlinear MPC
* The number of trials or experiment runs to perform: `num_trials`.
* To allow for running batches of experiments then collecting all of their data, the number of completed experiments can be specified in `trials_offset`.
    * `num_trials = 2`, `trials_offset = 0` generates runs stored in folders `000000000001, 000000000002`.
    * `num_trials = 3`, `trials_offset = 1` generates runs stored in folders `000000000002, 000000000003, 000000000004`.
    * Existing folders are overwritten.
* To generate new experiments set `run_flag` to `True`. 
    * To collect the data of already existing experiments:
        * Set `num_trials` to the available experiments
        * Set `trials_offset` to the number of the first experiment
        * Set `run_flag` to `False`
Note: all integers between `trials_offset` and `trials_offset + num_trials` must exist. Otherwise, an error is thrown.

### Saved Data Directory
`saved_data`: Preferable directory to store data.
### Monte Carlo Results Directory
`monte_carlo`: Directory where Monte Carlo results are stored.

## Using the Package
To use the package follow these steps.
1. Update the config file (`config.py`).
2. Run the RANS-RRT* file (`rans_rrtstar.py`).
    * this generates the trajectory and saves two files
3. Copy the name of the node list (e.g. `NodeListData_v2_0_1623778273`).
4. Update the `filename` variable in `main` of `get_opt_paths.py` with the copied node list name. 
    * this extract the optimal trajectory and shortens it (in time) and saves six files
5. Copy the name of the trajectory's inputs (e.g. `OptTraj_short_v2_0_1623778273_inputs` or `OptTraj_v2_0_1623778273_inputs`).
6. Update the `input_file` variable in `main` of `mote_carlo.py` with the copied input file name.
7. Update the remaining configuraition variables in `mote_carlo.py`
    * `UNIQUE_EXP_NUM` at the top of the script
    * `input_file`, `noise_dist`, `controller_str_list`, `num_trials`, `trials_offset`, and `run_flag` in `main`
8. Run the Monte-Carlo script `mote_carlo.py`,
    * this will generate the experiment runs in 
      `monte_carlo/env<UNIQUE_EXP_NUM>/<trials_offset+1>` through
      `monte_carlo/env<UNIQUE_EXP_NUM>/<trials_offset+num_trials>`.
    * it will also save the trajectory images in `monte_carlo/path_plots/env<UNIQUE_EXP_NUM>`.
    * statistics about the results are printed.
    
## License
See LICENSE file.
