
% This script provides a MATLAB interface to load the RANS-RRT* tree
% generated by the python scripts and use it.

close all
clear all
clc

%% Parameters (these must match the python config params)
N = 30; % NLP steering horizon
n = 3; % number of states
m = 2; % number of inputs
dt = 0.2; % discrete time step

% this is the csv equivalent of the pickle file generated by RANS-RRT* 
% this can also be generated by the rans_rrtstar.py script using the
% "main_from_data" function which requires setting `run_rrt` in "main" to
% False and using the pickle file name in "main_from_data" then running the
% script.
% NOTE: this script must be in this matlab directory
NodeListData_filename = 'NodeListData_v2_0_1624832981.csv'; 


%% Load the data

data = readtable(NodeListData_filename); % read csv file
data = table2array(data); % convert csv table to array

% extract the data
parent = data(:,1); % list of parent of each node 
X = data(:,2:1+(N+1)*n); % trajectory states at each node
U = data(:, 2+(N+1)*n:end); % control inputs for each trajectory at each node
num_trajectories = size(X,1); % number of trajectories in the tree




















