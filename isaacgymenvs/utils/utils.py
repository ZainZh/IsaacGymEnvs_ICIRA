# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# python

import numpy as np
import torch
import random
import os
import h5py

def set_np_formatting():
    """ formats numpy print """
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def set_seed(seed, torch_deterministic=False):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed

# add
class HDF5DatasetWriter():
    def __init__(self, outputPath, action_size=18, obs_size=74,bufSize=1000, maxSize=None):
        # 如果输出文件路径存在，提示异常
        # if os.path.exists(outputPath):
        #     raise ValueError("The supplied 'outputPath' already exists and cannot be overwritten. Manually delete the file before continuing", outputPath)

        self._action_size = action_size
        self._obs_size = obs_size

        # 构建两种数据，一种用来存储图像特征一种用来存储标签
        self.db = h5py.File(outputPath, "w")
        self.actions = self.db.create_dataset('actions', (1, self._action_size), maxshape=(maxSize, self._action_size), dtype="float")
        self.observations = self.db.create_dataset('observations', (1, self._obs_size), maxshape=(maxSize, self._obs_size), dtype="float")
        self.next_observations = self.db.create_dataset('next_observations', (1, self._obs_size), maxshape=(maxSize, self._obs_size),
                                                        dtype="float")
        self.rewards = self.db.create_dataset('rewards', (1, 1), maxshape=(maxSize, 1), dtype="float")
        self.dones = self.db.create_dataset("dones", (1, 1), maxshape=(maxSize, 1), dtype="i8")

        # 设置buffer大小，并初始化buffer
        self.bufSize = bufSize
        self.buffer = {"actions": [], "observations": [],
                       "next_observations": [], "rewards": [],
                       "dones": [], }
        self.idx = 0  # 用来进行计数

    def add(self, obs, action, reward, next_obs, done):
        # if isinstance(obs, torch.Tensor):
        #     _action=action.cpu().clone().view(-1, self._action_size).numpy()
        #     _obs=obs.cpu().clone().view(-1, self._obs_size).numpy()
        #     _reward=reward.cpu().clone().view(-1, 1).numpy()
        #     _next_obs=next_obs.cpu().clone().view(-1, self._obs_size).numpy()
        #     _done=done.cpu().clone().view(-1, 1).numpy()
        self.buffer["actions"].extend(action)
        self.buffer["observations"].extend(obs)
        self.buffer["next_observations"].extend(next_obs)
        self.buffer["rewards"].extend(reward)
        self.buffer["dones"].extend(done)

        # 查看是否需要将缓冲区的数据添加到磁盘中
        if len(self.buffer["actions"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # 将buffer中的内容写入磁盘之后重置buffer
        i = self.idx + len(self.buffer["actions"])
        if i >= len(self.actions):
            self.actions.resize((i, self._action_size))
            self.observations.resize((i, self._obs_size))
            self.next_observations.resize((i, self._obs_size))
            self.rewards.resize((i, 1))
            self.dones.resize((i, 1))
        self.actions[self.idx:i] = self.buffer["actions"]
        self.observations[self.idx:i] = self.buffer["observations"]
        self.next_observations[self.idx:i] = self.buffer["next_observations"]
        self.rewards[self.idx:i] = self.buffer["rewards"]
        self.dones[self.idx:i] = self.buffer["dones"]
        self.idx = i
        self.buffer = {"actions": [], "observations": [],
                       "next_observations": [], "rewards": [],
                       "dones": [], }
        print('hdf5 data flush, {} rows'.format(i))

    def close(self):
        if len(self.buffer["actions"]) > 0:  # 查看是否缓冲区中还有数据
            self.flush()
        print('total length:', len(self.actions))
        self.db.close()
        print('hdf5 save success')


# EOF
