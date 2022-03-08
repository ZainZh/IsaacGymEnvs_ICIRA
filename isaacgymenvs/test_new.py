import argparse

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from tasks.dual_franka import *

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.rlgames_utils import get_rlgames_env_creator

from utils.utils import set_np_formatting, set_seed

import math
import numpy as np
import torch


## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

# global test args
args = gymutil.parse_arguments(description="Franka Tensor OSC Example",
                                custom_parameters=[
                                    {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
                                    {"name": "--pos_control", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Trace circular path in XZ plane"},
                                    {"name": "--orn_control", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Send random orientation commands"}])

def myparser(args,cfg_path):
    temp=OmegaConf.load(cfg_path)
    args=vars(args)
    res=list()
    for key,value in args.items():
        if key not in temp.keys():
            pass
        else:
            if key == 'physics_engine':
                res.append('physics_engine=physx')
            else:
                res.append(str(key)+'='+str(value))
    res.append('pipeline=None')     # force pipeline=None 
    return res

def get_cfg():
    # override
    res=myparser(args,'./cfg/config.yaml')
    from hydra import compose, initialize
    initialize(config_path="cfg")
    cfg = compose(config_name="config",overrides=res)

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    # add all other args that do not override
    cfg=dict(cfg)
    cfg.update(vars(args))
    cfg = argparse.Namespace(**cfg)

    return cfg

class DualFrankaTest(DualFranka):
    def __init__(self, cfg, sim_device, graphics_device_id, headless,sim_params):
        self.sim_params=sim_params
        super().__init__(cfg, sim_device, graphics_device_id, headless)

    def set_viewer(self):
        """Create the viewer."""
        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # Point camera at middle env
            cam_pos = gymapi.Vec3(4, 3, 3)
            cam_target = gymapi.Vec3(-4, -3, 0)
            middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        

if __name__ == "__main__":
    # parse from default config
    cfg=get_cfg()

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
    env=DualFrankaTest(cfg=omegaconf_to_dict(cfg.task),
            sim_device=cfg.sim_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=False,
            sim_params=sim_params)

    while not env.gym.query_viewer_has_closed(env.viewer):
        # Step the physics
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        # Step rendering
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(env.viewer, env.sim, False)
        env.gym.sync_frame_time(env.sim)
        print(env.compute_reward(''))
    print("Done")

    env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)