from scripts.rans_rrtstar import make_tree, load_and_plot_tree
from scripts.opt_path import opt_and_short_traj
from scripts.monte_carlo import monte_carlo_main

from filesearch import get_timestr
from config import SAVEPATH


# Build the RANS-RRT* tree
filename = make_tree()
load_and_plot_tree(filename)

# Extract the optimal trajectory from the tree and shorten it
opt_and_short_traj(filename, SAVEPATH, plot_opt_path=True, plot_short_opt_path=True)

# Run Monte Carlo simulations on the shortened reference trajectory
timestr = get_timestr(filename=filename)
monte_carlo_main(timestr, draft=True)

# TODO add programmatic plotting functionality - currently you have to run plotting.py manually
