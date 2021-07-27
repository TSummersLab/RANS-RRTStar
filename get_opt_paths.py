#!/usr/bin/env python3
"""
Changelog:
New is v1_0:
- Add script

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
Github:
@The-SS

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Find the optimal path generated by RRT* and the shortened version of it (high-level wrapper of `script/opt_path.py`

Tested platform:
- Python 3.6.9 on Ubuntu 18.04 LTS (64 bit)

"""
import sys

sys.path.insert(0, 'scripts')
from scripts.opt_path import opt_and_short_traj
import os
import numpy as np
from rans_rrtstar import DR_RRTStar_Node

import config

SAVEPATH = config.SAVEPATH  # path where RRT* data is located and where this data will be stored


#####################################################NodeListData_v1_0_1607441929##########################
###############################################################################
def main():
    filename = "NodeListData_v2_0_1624832981"  # name of RRT* pickle file to process
    v_max = 0.5
    omega_max = np.pi
    num_states = 3
    num_controls = 2
    opt_and_short_traj(filename, SAVEPATH, v_max, omega_max, num_states, num_controls,
                       save_opt_path=True, plot_opt_path=True, save_opt_path_plot=True,
                       save_short_opt_path=True, plot_short_opt_path=True, save_short_opt_path_plot=True)


if __name__ == "__main__":
    main()
