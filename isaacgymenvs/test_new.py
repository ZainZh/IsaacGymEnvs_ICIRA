import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.rlgames_utils import get_rlgames_env_creator

from utils.utils import set_np_formatting, set_seed

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
from tasks.dual_franka import *


## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)


def get_cfg():
    from hydra import compose, initialize
    initialize(config_path="cfg")
    cfg = compose(config_name="config")

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    return cfg_dict

class DualFrankaTest(DualFranka):
    def __init__(self, cfg, sim_device, graphics_device_id, headless,sim_params):
        self.sim_params=sim_params
        super().__init__(cfg, sim_device, graphics_device_id, headless)
        
        

if __name__ == "__main__":
    # parse from default config
    cfg=get_cfg()

    # override
    args = gymutil.parse_arguments(description="Franka Tensor OSC Example",
                                    custom_parameters=[
                                        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
                                        {"name": "--pos_control", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Trace circular path in XZ plane"},
                                        {"name": "--orn_control", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Send random orientation commands"}])
    
    cfg.update(args)

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Y
    sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.up_axis = gymapi.UP_AXIS_Y
    else:
        raise Exception("This example can only be used with PhysX")

    sim_params.use_gpu_pipeline = False

    # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
    # We use the helper function here to specify the environment config.
    env=DualFrankaTest(cfg=cfg,
            sim_device=cfg.sim_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless,
            sim_params=sim_params)

    while not env.gym.query_viewer_has_closed(env.viewer):
        # Step the physics
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        # Step rendering
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(env.viewer, env.sim, False)
        env.gym.sync_frame_time(env.sim)
    print("Done")

    env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)