import h5py
import torch
import numpy as np

date_file = h5py.File('./test_save/hdf5/0725-06-38-14.hdf5', 'r')
date_file_old = h5py.File('./reset_buffer/replay_buff1.hdf5', 'r')
print(list(date_file.keys()))
print(list(date_file_old.keys()))
data_obs = torch.tensor(np.array(date_file['observations_left']), dtype=torch.float)
data_actions = torch.tensor(np.array(date_file['actions_left']), dtype=torch.float)
data_rewards = torch.tensor(np.array(date_file['rewards_left']), dtype=torch.float)
data_next_obs = torch.tensor(np.array(date_file['next_observations_left']), dtype=torch.float)
data_dones = torch.tensor(np.array(date_file['dones_left']), dtype=torch.float)