import h5py
import torch
import numpy as np

date_file = h5py.File('test_save/hdf5/08-01-06_21_47.hdf5', 'r')

print(list(date_file.keys()))

data_obs = torch.tensor(np.array(date_file['observations_left']), dtype=torch.float)
data_actions = torch.tensor(np.array(date_file['actions_left']), dtype=torch.float)
data_rewards = torch.tensor(np.array(date_file['rewards_left']), dtype=torch.float)
data_next_obs = torch.tensor(np.array(date_file['next_observations_left']), dtype=torch.float)
data_dones = torch.tensor(np.array(date_file['dones_left']), dtype=torch.float)