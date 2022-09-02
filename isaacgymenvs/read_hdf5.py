import h5py
import torch
import numpy as np

date_file = h5py.File('test_save/hdf5/09-02-12_03_53.hdf5', 'r')
cup_npy = np.load('test_save/cup.npy')
spoon_npy = np.load('test_save/spoon.npy')
cup_pos = cup_npy[:, :7]
spoon_pos = spoon_npy[:, :7]
cup_interpos = np.zeros([50, 7], dtype=float)
spoon_interpos = np.zeros([50, 7], dtype=float)
j = 0
for i in range(int(cup_pos.shape[0] / 10)):
    cup_interpos[i, :] = cup_pos[j, :]
    spoon_interpos[i, :] = spoon_pos[j, :]
    j += 10

total_stage = np.zeros([500, 9, 2], dtype=float)
spoon_gripper = np.zeros([500, 2], dtype=float)

spoon_gripper[156:, :] = [0.0035, 0.0035]
spoon_gripper[:156, :] = [0.04, 0.04]

cup_gripper = np.zeros([500, 2], dtype=float)
cup_gripper[156:, :] = [0.024, 0.024]
cup_gripper[:156, :] = [0.04, 0.04]

franka1_pos = np.hstack((cup_pos, cup_gripper))
franka1_pos[156, :] = np.array([-0.0723, 0.8304, 0.5483, 0.8606, 0.3482, 0.3238, -0.1825, 0.0400, 0.0400])

franka_pos = np.hstack((spoon_pos, spoon_gripper))
franka_pos[156, :] = np.array([-0.1780, 0.9759, -0.4035, -0.4742, -0.1043, -0.7933, -0.3672, 0.0300, 0.0300])
total_stage[:, :, 0] = franka_pos
total_stage[:, :, 1] = franka1_pos
print(list(date_file.keys()))

data_obs = torch.tensor(np.array(date_file['observations_left']), dtype=torch.float)
data_actions = torch.tensor(np.array(date_file['actions_left']), dtype=torch.float)
data_rewards = torch.tensor(np.array(date_file['rewards_left']), dtype=torch.float)
data_next_obs = torch.tensor(np.array(date_file['next_observations_left']), dtype=torch.float)
data_dones = torch.tensor(np.array(date_file['dones_left']), dtype=torch.float)
