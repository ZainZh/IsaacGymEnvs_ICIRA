# train.py
# Script to train policies in Isaac Gym
#
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

import isaacgym

import os
import shutil

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from utils.utils import set_np_formatting, set_seed

# from isaacgymenvs.learning import amp_continuous
# from isaacgymenvs.learning import amp_players
# from isaacgymenvs.learning import amp_models
# from isaacgymenvs.learning import amp_network_builder


## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    if cfg.dataset:
        cfg.dataset = to_absolute_path(cfg.dataset)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
    # We use the helper function here to specify the environment config.
    create_rlgpu_env = get_rlgames_env_creator(
        omegaconf_to_dict(cfg.task),
        cfg.task_name,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        multi_gpu=cfg.multi_gpu,
    )

    # register the rl-games adapter to use inside the runner
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    })

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        # runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        # runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        # runner.model_builder.model_factory.register_builder('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))  
        # runner.model_builder.network_factory.register_builder('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # convert CLI arguments into dictionory
    # create runner and set the settings
    runner = build_runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    experiment_dir = os.path.join('runs', cfg.train.params.config.name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    if cfg.task_name == 'DualFranka':
        loacl_dir = os.getcwd()
        Dualfranka_dir = os.path.join(loacl_dir, 'tasks/dual_franka.py')
        A2Ccommon_dir = os.path.join(loacl_dir, 'rl_games/common/a2c_common.py')
        A2Ccontinuous_dir = os.path.join(loacl_dir, 'rl_games/algos_torch/a2c_continuous.py')
        expectdual_dir = os.path.join(loacl_dir, experiment_dir, 'dual_franka.py')
        expectcomm_dir = os.path.join(loacl_dir, experiment_dir, 'a2c_common.py')
        expectcont_dir = os.path.join(loacl_dir, experiment_dir, 'a2c_continuous.py')
        shutil.copyfile(f'{Dualfranka_dir}', f"{expectdual_dir}")
        shutil.copyfile(f'{A2Ccommon_dir}', f"{expectcomm_dir}")
        shutil.copyfile(f'{A2Ccontinuous_dir}', f"{expectcont_dir}")
    if cfg.train.params.algo.name == 'cql' or cfg.train.params.algo.name == 'sac':
        runner.run({
            'train': not cfg.test,  # decide train or play in cfg
            'play': cfg.test,
            'checkpoint': cfg.checkpoint,  # checkpoint arg
            'sigma': cfg.train.params.config.sigma,  # torch_runner line 26
            'dataset': cfg.dataset,
        })
    else:
        runner.run({
            'train': not cfg.test,  # decide train or play in cfg
            'play': cfg.test,
            'checkpoint': cfg.checkpoint,  # checkpoint arg
            # 'sigma': cfg.train.params.config.sigma,  # torch_runner line 26
        })


if __name__ == "__main__":
    launch_rlg_hydra()
