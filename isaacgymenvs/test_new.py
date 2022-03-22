
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from tasks.dual_franka import *

import argparse
import configparser
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from utils.utils import set_np_formatting, set_seed
import math
import random
import numpy as np
from typing import Dict,Tuple,List

torch.set_printoptions(precision=4, sci_mode=False)
test_config = configparser.ConfigParser()
test_config.read('test_config.ini')
franka_cfg_path = test_config['PRESET'].get('franka_cfg_path', './cfg/config.yaml')
print_mode = test_config['PRESET'].getint('print_mode',2)
target_data_path = test_config["SIM"].get('target_data_path', None)
enable_dof_target = test_config["DEFAULT"].getboolean('enable_dof_target', False)
control_k = test_config["SIM"].getfloat('control_k', 1.0)
damping = test_config["SIM"].getfloat('damping', 0.05)
if target_data_path is None:
    enable_dof_target = False


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
# self define functions
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
    res.append('task=DualFranka')     
    return res

def get_cfg(path):
    # override
    res=myparser(args,path)
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

def save():
    print("here save!")
    # TODO

def load_target_ee(filepath):
    # read target ee
    with open(filepath, "r") as f:
        txtdata = f.read()
    import re
    x=txtdata.split("]")
    res=[]
    for i in x:
        tmp=re.findall(r"-?\d+\.?\d*", i)
        if len(tmp) == 9:
            res.append(tmp)
        elif len(tmp) == 0:
            pass
        else:
            raise Exception('data format error')
    if len(res) % 2:
        raise Exception('data format error')

    target=np.array(res).astype(float).reshape(-1,2,9)
    
    return torch.from_numpy(target)

def parse_reward_detail(dictobj:Dict):
    for k,v in dictobj.items():
        for k1,v1 in v.items():
            if isinstance(v1,tuple):
                new_list = list()
                for obj in v1:
                    if isinstance(obj,torch.Tensor):
                        new_list.append(obj.tolist()[0] if obj.shape==1 else obj.tolist())
                    else:
                        new_list.append(obj)
                dictobj[k][k1] = tuple(new_list)
            elif isinstance(v1,torch.Tensor):
                dictobj[k][k1] = v1.tolist()[0] if v1.shape==1 else v1.tolist()

    return dictobj

def print_detail_clearly(dictobj):
    print('rew_details')
    obj = parse_reward_detail(dictobj)
    for k,v in obj.items():
        print("\033[1;32m"+str(k)+"\033[0m")
        if isinstance(v,Dict):
            findoprint(v)
        else:
            print(str(k)+": "+str(v))
    
def findoprint(dictobj):
    for k,v in dictobj.items():
        if isinstance(v,Dict):
            findoprint(v)
        else:
            print(str(k)+": "+str(v))


total_print_mode = 3
def print_state(if_all=False):
    if print_mode >= 1 or if_all==True:
        print('actions', pos_action.numpy())
        franka_dof = gymtorch.wrap_tensor(env.gym.acquire_dof_state_tensor(env.sim))[:,0].view(2,-1)
        print('franka_dof', franka_dof)
        gripper_dof = franka_dof[:,-2:]
        ee_pose = torch.cat((env.rigid_body_states[:, env.hand_handle][:,0:7],env.rigid_body_states[:, env.hand_handle_1][:,0:7]))
        print('ee_pose&gripper',torch.cat((ee_pose,gripper_dof), dim=1))
        print('obs-',env.compute_observations())
        print('rew-',env.compute_reward())
    
    if print_mode >= 2 or if_all==True:
        print_detail_clearly(env.reward_dict)

    # print reset env_ids
    if print_mode >= 1 or if_all==True:
        env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.shape[0] != 0:
            print('trigger reset:',env_ids.numpy())
    
# camera pos
cam_switch = 0
cam_pos = [
    [[0.66, 0.75, 1.20],[-2.80, -0.83, -3.14]], # from RF
    [[0.66, 0.75, -1.20],[-2.80, -0.83, 3.14]], # from LF
    [[-1.77, 0.81, -0.17],[3.14, -0.80, -0.7]], # from behind
]

# DualFranka class for test
class DualFrankaTest(DualFranka):
    def __init__(self, cfg, sim_device, graphics_device_id, headless,sim_params):
        self.sim_params=sim_params
        super().__init__(cfg, sim_device, graphics_device_id, headless)
        # copy from line175
        # self.franka_dof_stiffness = torch.tensor([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        # self.franka_dof_damping = torch.tensor([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)


    def set_viewer(self):
        """Create the viewer."""
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a vienv.viewerewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "reset")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_P, "change_print_state")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_S, "save")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_T, "print_once")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_C, "switch_cam_view")

            # set camera view
            self.cam_view_switch(cam_pos[cam_switch])
    
    def compute_reward(self):
        # no action penalty in test
        self.actions=torch.zeros(self.cfg["env"]["numActions"]).to(self.device)
        super().compute_reward()
        return self.rew_buf
    
    def cam_view_switch(self,vec):
        # Point camera at middle env
        num_per_row = int(math.sqrt(self.num_envs))
        cam_pos = gymapi.Vec3(*vec[0])
        cam_target = gymapi.Vec3(*vec[1])
        middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

# calculation
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)    


def control_ik(dpose,jacobian):
    # solve damped least squares
    j_eef_T = torch.transpose(jacobian, 1, 2)
    lmbda = torch.eye(6, device=env.device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(jacobian @ j_eef_T + lmbda) @ dpose).view(env.num_envs, 7)
    return u


if __name__ == "__main__":
    # parse from default config
    cfg=get_cfg(franka_cfg_path)

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

    env=DualFrankaTest(cfg=omegaconf_to_dict(cfg.task),
            sim_device=cfg.sim_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=False,
            sim_params=sim_params)

    # inital values
    t = 0
    left_action = torch.zeros_like(env.franka_dof_state_1[...,0]).squeeze(-1)   # only need [...,0]->position, 1 for velocity
    right_action = torch.zeros_like(env.franka_dof_state[...,0]).squeeze(-1)
    pos_action = torch.zeros_like(torch.cat((right_action,left_action), dim=0))
    if enable_dof_target:
        now_stage = 0
        target_pose = load_target_ee(target_data_path).to(env.device)

    while not env.gym.query_viewer_has_closed(env.viewer):
        
        # Get input actions from the viewer and handle them appropriately
        for evt in env.gym.query_viewer_action_events(env.viewer):
            if evt.value > 0:
                if evt.action == "reset":
                    # need to disable pose override in viewer
                    print('Reset env')
                    env.reset_idx(torch.arange(env.num_envs, device=env.device))
                elif evt.action == "save":
                    print_state()
                    save()
                elif evt.action == "change_print_state":
                    print("Change print mode")
                    print_mode += 1
                    if print_mode >= total_print_mode:
                        print("Stop printing")
                        print_mode = 0
                elif evt.action == "print_once":
                    print("Print once",random.randint(0,100))
                    print_state(if_all=True)
                elif evt.action == "switch_cam_view":
                    print("Switch view")
                    cam_switch += 1
                    if cam_switch >= len(cam_pos):
                        cam_switch = 0
                    env.cam_view_switch(cam_pos[cam_switch])
        
        # Step the physics
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)

        if enable_dof_target:
            # refresh tensors
            env.gym.refresh_actor_root_state_tensor(env.sim)
            env.gym.refresh_dof_state_tensor(env.sim)
            env.gym.refresh_net_contact_force_tensor(env.sim)
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            env.gym.refresh_jacobian_tensors(env.sim)

            ## Calculation here
            # get jacobian tensor
            # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
            _jacobian = env.gym.acquire_jacobian_tensor(env.sim, "franka")
            _jacobian_1 = env.gym.acquire_jacobian_tensor(env.sim, "franka1")
            jacobian_left = gymtorch.wrap_tensor(_jacobian_1)
            jacobian_right = gymtorch.wrap_tensor(_jacobian)

            # get link index of panda hand, which we will use as end effector
            # franka_link_dict = env.gym.get_asset_rigid_body_dict(franka_asset)
            # franka_hand_index = franka_link_dict["panda_hand"]
            franka_hand_index = 8   # just set to 8 instead

            # jacobian entries corresponding to franka hand
            j_eef_left = jacobian_left[:, franka_hand_index - 1, :, :7]
            j_eef_right = jacobian_right[:, franka_hand_index - 1, :, :7]

            # decide goal(target)
            now_target = target_pose[now_stage, ...]

            left_hand_pos = env.rigid_body_states[:, env.hand_handle_1][:, 0:3]
            left_goal_pos = now_target[1, 0:3]
            left_hand_rot = env.rigid_body_states[:, env.hand_handle_1][:, 3:7]
            left_goal_rot = now_target[1, 3:7]
            
            right_hand_pos = env.rigid_body_states[:, env.hand_handle][:, 0:3]
            right_goal_pos = now_target[0, 0:3]
            right_hand_rot = env.rigid_body_states[:, env.hand_handle][:, 3:7]
            right_goal_rot = now_target[0, 3:7]

            # compute position and orientation error
            left_pos_err = left_goal_pos - left_hand_pos
            left_orn_err = orientation_error(left_goal_rot, left_hand_rot)
            left_dpose = torch.cat([left_pos_err, left_orn_err], -1).unsqueeze(-1)
            right_pos_err = right_goal_pos - right_hand_pos
            right_orn_err = orientation_error(right_goal_rot, right_hand_rot)
            right_dpose = torch.cat([right_pos_err, right_orn_err], -1).unsqueeze(-1)

            # left_dof_pos = env.franka_dof_state_1[..., 0].view(env.num_envs, 9, 1)
            # left_dof_vel = env.franka_dof_state_1[..., 1].view(env.num_envs, 9, 1)
            # right_dof_pos = env.franka_dof_state[..., 0].view(env.num_envs, 9, 1)
            # right_dof_vel = env.franka_dof_state_1[..., 1].view(env.num_envs, 9, 1)

            # body ik, relative control
            left_action[:, :7] = control_k * control_ik(left_dpose,j_eef_left)
            right_action[:, :7] = control_k * control_ik(right_dpose,j_eef_right)
            # gripper actions
            left_action[:, 7:9] = now_target[1, 7:9]
            right_action[:, 7:9] = now_target[0, 7:9]
            # merge two franka
            pos_action = torch.cat((right_action,left_action), dim=0)

            # Deploy actions
            # just comment this line if don't need action
            env.pre_physics_step(pos_action.view(env.num_envs,-1))
        
        # Step rendering
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(env.viewer, env.sim, False)
        env.gym.sync_frame_time(env.sim)
        
        # print obs/reward
        t += 1
        if t % 50 == 0:
            t = 0
            if print_mode != 0:
                print_state()
    
    print("Done")
    env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)

