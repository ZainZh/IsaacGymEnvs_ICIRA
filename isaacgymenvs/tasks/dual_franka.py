import math
import numpy as np
import os
import torch
import h5py
from torch import Tensor
import random
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.base.vec_task import VecTask
from typing import Dict, List, Tuple, Union

shelf_as_box = False
spoon_as_box = False
turn_spoon = False


class DualFranka(VecTask):
    # define the config
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.num_Envs = self.cfg["env"]["numEnvs"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.lift_reward_scale = self.cfg["env"]["liftRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.stage2_3_scale = self.cfg["env"]["stage2_3_scale"]
        self.num_agents = self.cfg["env"]["numAgents"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.ResetFromReplay = self.cfg["env"]["ResetFromReplay"]
        self.stage2begin = self.cfg['env']['stage2begin']
        self.ReadExpertData = self.cfg['env']['ReadExpertData']
        self.damping = self.cfg['env']['Damping']
        self.UsingIK = self.cfg['env']['UsingIK']
        self.High_stiffness = self.cfg['env']['HighStiffness']
        self.LocationNoise = self.cfg['env']['LocationNoise']
        self.up_axis = "y"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.

        # prop dimensions
        self.prop_width = 0.06
        self.prop_height = 0.06
        self.prop_length = 0.06
        self.prop_spacing = 0.09
        self.index_of_simulation = 0
        self.gripped = torch.zeros((1, self.num_Envs))
        self.gripped_1 = torch.zeros((1, self.num_Envs))
        # num_obs = 42
        # num_acts = 18
        actors_per_env = 6
        self.cfg["env"]["numObservations"] = 74
        self.cfg["env"]["numActions"] = 18
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless)
        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # get the position and orientation tensor (x,y,z,(0,0,0,1) )
        self.root_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_Envs, actors_per_env, 13)
        # self.root_states=self.root_tensor
        # self.saved_root_tensor = self.root_tensor.clone()
        self.cup_positions = self.root_tensor[..., 1, 0:3]
        self.cup_orientations = self.root_tensor[..., 1, 3:7]
        self.cup_linvels = self.root_tensor[..., 1, 7:10]
        self.cup_angvels = self.root_tensor[..., 1, 10:13]
        self.spoon_positions = self.root_tensor[..., 2, 0:3]
        self.spoon_orientations = self.root_tensor[..., 2, 3:7]
        self.spoon_linvels = self.root_tensor[..., 2, 7:10]
        self.spoon_angvels = self.root_tensor[..., 2, 10:13]
        self.shelf_positions = self.root_tensor[..., 3, 0:3]
        self.shelf_orientations = self.root_tensor[..., 3, 3:7]
        self.shelf_linvels = self.root_tensor[..., 3, 7:10]
        self.shelf_angvels = self.root_tensor[..., 3, 10:13]
        self.box_positions = self.root_tensor[..., 4, 0:3]
        self.box_orientations = self.root_tensor[..., 4, 3:7]
        self.box_linvels = self.root_tensor[..., 4, 7:10]
        self.box_angvels = self.root_tensor[..., 4, 10:13]
        self.table_positions = self.root_tensor[..., 5, 0:3]
        self.table_orientations = self.root_tensor[..., 5, 3:7]
        self.table_linvels = self.root_tensor[..., 5, 7:10]
        self.table_angvels = self.root_tensor[..., 5, 10:13]
        self.control_k = 0.6

        # self.all_actor_indices = torch.arange(actors_per_env * self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs, actors_per_env)

        # create some wrapper tensors for different slices
        # set default pos (seven links and left and right hand)
        self.franka_default_dof_pos = to_torch([0.3863, 0.5062, -0.1184, -0.6105, 0.023, 1.6737, 0.9197, 0.04, 0.04],
                                               device=self.device)
        self.franka_default_dof_pos_1 = to_torch([-0.5349, 0, 0.1401, -1.7951, 0.0334, 3.2965, 0.6484, 0.04, 0.04],
                                                 device=self.device)
        self.franka_default_dof_pos_stage2 = to_torch(
            [-0.3545, 0.6990, 0.2934, 0.7159, -0.1442, 0.3583, 0.5817, 0.04, 0.04],
            device=self.device)
        self.franka_default_dof_pos_1_stage2 = to_torch(
            [-0.4901, 0.5585, -0.3762, -0.5309, -0.1273, -0.7353, -0.4016, 0.04, 0.04],
            device=self.device)
        self.franka_default_dof_pos_2 = to_torch([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                 device=self.device)
        self.franka_default_dof_pos_3 = to_torch([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                 device=self.device)
        self.franka_default_dof_pos_4 = to_torch([0.3269, 0.2712, 0.0, -1.0166, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                 device=self.device)
        self.franka_default_dof_pos_5 = to_torch([-0.4457, -0.0542, 0.0, -1.7401, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                 device=self.device)
        # order of load asset:franka franka1 cup spoon shelf table
        # get every dof state (pos and vel)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:,
                                3 + self.num_franka_dofs:3 + 2 * self.num_franka_dofs]
        self.franka_dof_state_1 = self.dof_state.view(self.num_envs, -1, 2)[:, 3:3 + self.num_franka_dofs]

        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]
        self.franka_dof_pos_1 = self.franka_dof_state_1[..., 0]
        self.franka_dof_vel_1 = self.franka_dof_state_1[..., 1]

        # detect collision (detect the force of every link)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        # get every actor's rigid body info(like dof or num)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.curi_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.curi_dof_targets_spoon = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float,
                                                  device=self.device)
        self.curi_dof_targets_cup = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.global_indices = torch.arange(self.num_envs * actors_per_env, dtype=torch.int32, device=self.device).view(
            self.num_envs,
            -1)

        if self.ReadExpertData:
            with h5py.File('replay_buffer/replaybuffer_fullstage_curi.hdf5  ', 'r') as hdf:
                self.data_actions_left = torch.tensor(np.array(hdf['actions_left']), dtype=torch.float,
                                                      device=self.device)
                # left obs
                self.data_obs_left = torch.tensor(np.array(hdf['observations_left']), dtype=torch.float,
                                                  device=self.device)
                # right action
                self.data_actions_right = torch.tensor(np.array(hdf['actions_right']), dtype=torch.float,
                                                       device=self.device)
                # right obs
                self.data_obs_right = torch.tensor(np.array(hdf['observations_right']), dtype=torch.float,
                                                   device=self.device)

            # rand_idx = random.randrange(0, dataset1.shape[0], 1)

        if self.ResetFromReplay:
            self.reset_idx_replay_buffer(torch.arange(self.num_envs, device=self.device))
        else:
            self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Y
        self.sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def control_ik(self, dpose, jacobian):
        # solve damped least squares
        j_eef_T = torch.transpose(jacobian, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (self.damping ** 2)
        u = (j_eef_T @ torch.inverse(jacobian @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, spacing, num_per_row):
        # create environment of simulation (the location of every items and some details)

        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        cup_asset_file = 'urdf/cup/urdf/cup.urdf'
        spoon_asset_file = 'urdf/spoon_new/urdf/spoon_new.urdf'
        shelf_asset_file = 'urdf/shelf/urdf/shelf.urdf'
        curi_asset_file = "urdf/curi/urdf/curi_pinocchio.urdf"

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.armature = 0.01
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)
        franka_asset_1 = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)
        curi_asset = self.gym.load_asset(self.sim, asset_root, curi_asset_file, asset_options)
        # load table, cup asset
        table_dims = gymapi.Vec3(1, 0.75, 1.5)
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        other_asset_options = gymapi.AssetOptions()
        cup_asset = self.gym.load_asset(self.sim, asset_root, cup_asset_file, other_asset_options)

        # load shelf and spoon
        box_dims = gymapi.Vec3(0.1, 0.04, 0.1)
        other_asset_options.fix_base_link = True
        shelf_asset = self.gym.load_asset(self.sim, asset_root, shelf_asset_file, other_asset_options)
        if shelf_as_box:
            shelf_box_dims = gymapi.Vec3(0.2, 0.1, 0.2)
            shelf_asset = self.gym.create_box(self.sim, shelf_box_dims.x, shelf_box_dims.y, shelf_box_dims.z,
                                              other_asset_options)
        box_asset = self.gym.create_box(self.sim, box_dims.x, box_dims.y, box_dims.z, other_asset_options)
        other_asset_options.fix_base_link = False
        spoon_asset = self.gym.load_asset(self.sim, asset_root, spoon_asset_file, other_asset_options)
        if spoon_as_box:
            spoon_box_dims = gymapi.Vec3(0.03, 0.03, 0.03)
            spoon_asset = self.gym.create_box(self.sim, spoon_box_dims.x, spoon_box_dims.y, spoon_box_dims.z,
                                              other_asset_options)

        if self.High_stiffness:
            franka_dof_stiffness = to_torch([1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6],
                                            dtype=torch.float,
                                            device=self.device)
        else:
            franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float,
                                            device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        self.num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        self.num_table_dofs = self.gym.get_asset_dof_count(table_asset)

        self.num_franka_bodies_1 = self.gym.get_asset_rigid_body_count(franka_asset_1)
        self.num_franka_dofs_1 = self.gym.get_asset_dof_count(franka_asset_1)

        self.num_cup_bodies = self.gym.get_asset_rigid_body_count(cup_asset)
        self.num_cup_dofs = self.gym.get_asset_dof_count(cup_asset)

        self.num_spoon_bodies = self.gym.get_asset_rigid_body_count(spoon_asset)
        self.num_spoon_dofs = self.gym.get_asset_dof_count(spoon_asset)

        self.num_shelf_bodies = self.gym.get_asset_rigid_body_count(shelf_asset)
        self.num_shelf_dofs = self.gym.get_asset_dof_count(shelf_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)
        print("num franka bodies_1: ", self.num_franka_bodies_1)
        print("num franka dofs_1: ", self.num_franka_dofs_1)
        print("num cup bodies: ", self.num_cup_bodies)
        print("num cup dofs: ", self.num_cup_dofs)
        print("num spoon bodies: ", self.num_cup_bodies)
        print("num spoon dofs: ", self.num_cup_dofs)
        print("num shelf bodies: ", self.num_shelf_bodies)
        print("num shelf dofs: ", self.num_shelf_dofs)
        print("num table bodies: ", self.num_table_bodies)
        print("num table dofs: ", self.num_table_dofs)

        # set franka dof properties
        franka_dof_props_1 = self.gym.get_asset_dof_properties(curi_asset)[3:3 + self.num_franka_dofs]
        franka_dof_props = self.gym.get_asset_dof_properties(curi_asset)[
                           3 + self.num_franka_dofs:3 + 2 * self.num_franka_dofs]
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self.camera_handles = [[]]
        self.norm_depth_image = [[]]
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            franka_dof_props_1['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
                franka_dof_props_1['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props_1['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0
                franka_dof_props_1['stiffness'][i] = 7000.0
                franka_dof_props_1['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200
        franka_dof_props_1['effort'][7] = 200
        franka_dof_props_1['effort'][8] = 200

        curi_dof_props = self.gym.get_asset_dof_properties(curi_asset)
        curi_dof_props[3:3 + self.num_franka_dofs] = franka_dof_props_1
        curi_dof_props[3 + self.num_franka_dofs:3 + 2 * self.num_franka_dofs] = franka_dof_props
        # create pose
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.5 * table_dims.y, 0.0)

        curi_pose = gymapi.Transform()
        curi_pose.p = gymapi.Vec3(-1.3, 0.0, 0.0)
        curi_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        box_pose = gymapi.Transform()
        box_pose.p.x = table_pose.p.x
        box_pose.p.y = table_pose.p.y + 0.5 * table_dims.y + 0.5 * box_dims.y
        box_pose.p.z = -0.29
        box_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.box_default_postion = to_torch([0.0, 0.77, 0.29], device=self.device)

        cup_pose = gymapi.Transform()
        cup_pose.p.x = table_pose.p.x
        cup_pose.p.y = box_pose.p.y + 0.5 * box_dims.y
        cup_pose.p.z = -0.29
        cup_pose.r = gymapi.Quat(0.0, -0.287, 0.0, 0.95793058)
        self.cup_default_postion = to_torch([0.0, 0.9, -0.29], device=self.device)

        shelf_pose = gymapi.Transform()
        shelf_pose.p.x = table_pose.p.x
        shelf_pose.p.y = table_pose.p.y + table_dims.y * 0.5
        shelf_pose.p.z = 0.29
        shelf_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.shelf_default_postion = to_torch([0.0, 0.75, 0.29], device=self.device)

        spoon_pose = gymapi.Transform()
        spoon_pose.p.x = table_pose.p.x
        spoon_pose.p.y = shelf_pose.p.y + 0.12
        spoon_pose.p.z = 0.29
        spoon_pose.r = gymapi.Quat(0.0, 0.0, 0.6, 0.707)
        self.spoon_default_postion = to_torch([0.0, 0.95, 0.29], device=self.device)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(curi_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(curi_asset)
        num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        num_cup_bodies = self.gym.get_asset_rigid_body_count(cup_asset)
        num_cup_shapes = self.gym.get_asset_rigid_shape_count(cup_asset)
        num_spoon_bodies = self.gym.get_asset_rigid_body_count(spoon_asset)
        num_spoon_shapes = self.gym.get_asset_rigid_shape_count(spoon_asset)
        num_shelf_bodies = self.gym.get_asset_rigid_body_count(shelf_asset)
        num_shelf_shapes = self.gym.get_asset_rigid_shape_count(shelf_asset)
        num_box_bodies = self.gym.get_asset_rigid_body_count(box_asset)
        num_box_shapes = self.gym.get_asset_rigid_shape_count(box_asset)
        max_agg_bodies = 2 * num_franka_bodies + num_spoon_bodies + num_table_bodies + num_cup_bodies + num_shelf_bodies + num_box_bodies
        max_agg_shapes = 2 * num_franka_shapes + num_spoon_shapes + num_table_shapes + num_cup_shapes + num_shelf_shapes + num_box_shapes

        self.curi = []
        self.table = []
        self.spoon = []
        self.cup = []
        self.shelf = []
        self.envs = []

        for i in range(self.num_envs):

            self.env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(self.env_ptr, max_agg_bodies, max_agg_shapes, True)

            curi_actor = self.gym.create_actor(self.env_ptr, curi_asset, curi_pose, "curi", i, 0)
            self.gym.set_actor_dof_properties(self.env_ptr, curi_actor, curi_dof_props)
            cup_actor = self.gym.create_actor(self.env_ptr, cup_asset, cup_pose, "cup", i, 0)
            spoon_actor = self.gym.create_actor(self.env_ptr, spoon_asset, spoon_pose, "spoon", i, 0)
            shelf_actor = self.gym.create_actor(self.env_ptr, shelf_asset, shelf_pose, "shelf", i, 0)
            box_actor = self.gym.create_actor(self.env_ptr, box_asset, box_pose, "box", i, 0)
            table_actor = self.gym.create_actor(self.env_ptr, table_asset, table_pose, "table", i, 0)
            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(self.env_ptr, max_agg_bodies, max_agg_shapes, True)
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(self.env_ptr)
            self.envs.append(self.env_ptr)
            self.curi.append(curi_actor)
            self.table.append(table_actor)
            self.cup.append(cup_actor)
            self.spoon.append(spoon_actor)
            self.shelf.append(shelf_actor)

        # add camera
        # self.set_camera()

        self.hand_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, curi_actor, "panda_right_link7")
        self.hand_handle_1 = self.gym.find_actor_rigid_body_handle(self.env_ptr, curi_actor, "panda_left_link7")
        self.table_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, table_actor, "table")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, curi_actor, "panda_right_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, curi_actor, "panda_right_rightfinger")
        self.lfinger_handle_1 = self.gym.find_actor_rigid_body_handle(self.env_ptr, curi_actor, "panda_left_leftfinger")
        self.rfinger_handle_1 = self.gym.find_actor_rigid_body_handle(self.env_ptr, curi_actor,
                                                                      "panda_left_rightfinger")
        self.cup_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, cup_actor, "base_link")
        self.spoon_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, spoon_actor, "base_link")
        self.shelf_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, shelf_actor, "base_link")
        self._curi_jacobian = self.gym.acquire_jacobian_tensor(self.sim, "curi")
        self.curi_jacobian = gymtorch.wrap_tensor(self._curi_jacobian)
        self.cup_curi_jacobian = self.curi_jacobian[:, 3:self.num_franka_dofs + 3, :, 3:self.num_franka_dofs + 3]
        self.spoon_curi_jacobian = self.curi_jacobian[:, self.num_franka_dofs + 3:2 * self.num_franka_dofs + 3, :,
                                   self.num_franka_dofs + 3:2 * self.num_franka_dofs + 3]
        franka_hand_index = 8  # just set to 8 instead
        # jacobian entries corresponding to franka hand
        self.j_eef_right = self.cup_curi_jacobian[:, franka_hand_index - 1, :, :7]
        self.j_eef_left = self.spoon_curi_jacobian[:, franka_hand_index - 1, :, :7]
        self.init_data()
        # self.gym.get_sim_rigid_body_states(self.sim,gymapi.STATE_ALL)

    def set_camera(self):
        # set up the camera props
        for i in range(self.num_Envs):
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.horizontal_fov = 75.0
            self.camera_props.width = 1920
            self.camera_props.height = 1080
            self.camera_props.enable_tensors = True
            handle = self.gym.create_camera_sensor(self.envs[i], self.camera_props)
            # handle=handle+1
            # set up the camera locations (middle point between two Frankas, the height is 1.3m)
            camera_location = gymapi.Vec3(-1, 1.3, -0.05)
            # set up the point that the camera looking at (middle point between the cup and the spoon)
            camera_location_view = gymapi.Vec3(-0.3, 0.5, 0.0)
            self.gym.set_camera_location(handle, self.envs[i], camera_location, camera_location_view)
            self.camera_handles[i].append(handle)
            self.camera_handles.append([])

    def init_data(self):  # define some init data for simulation
        # get franka data form
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.curi[0], "panda_right_link7")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.curi[0], "panda_right_leftfinger")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.curi[0], "panda_right_rightfinger")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        finger_pose.r = lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()

        grasp_pose_axis = 2  # forward axis = z
        franka_local_grasp_pose = hand_pose_inv * finger_pose
        franka_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))

        self.franka_local_grasp_pos = to_torch([franka_local_grasp_pose.p.x, franka_local_grasp_pose.p.y,
                                                franka_local_grasp_pose.p.z], device=self.device).repeat(
            (self.num_envs, 1))
        self.franka_local_grasp_rot = to_torch([franka_local_grasp_pose.r.x, franka_local_grasp_pose.r.y,
                                                franka_local_grasp_pose.r.z, franka_local_grasp_pose.r.w],
                                               device=self.device).repeat((self.num_envs, 1))

        # get franka_1 data form
        hand_1 = self.gym.find_actor_rigid_body_handle(self.envs[0], self.curi[0], "panda_left_link7")
        lfinger_1 = self.gym.find_actor_rigid_body_handle(self.envs[0], self.curi[0], "panda_left_leftfinger")
        rfinger_1 = self.gym.find_actor_rigid_body_handle(self.envs[0], self.curi[0], "panda_left_rightfinger")

        hand_pose_1 = self.gym.get_rigid_transform(self.envs[0], hand_1)
        lfinger_pose_1 = self.gym.get_rigid_transform(self.envs[0], lfinger_1)
        rfinger_pose_1 = self.gym.get_rigid_transform(self.envs[0], rfinger_1)

        finger_pose_1 = gymapi.Transform()
        finger_pose_1.p = (lfinger_pose_1.p + rfinger_pose_1.p) * 0.5
        finger_pose_1.r = lfinger_pose_1.r

        hand_pose_inv_1 = hand_pose_1.inverse()

        grasp_pose_axis_1 = 2
        franka_local_grasp_pose_1 = hand_pose_inv_1 * finger_pose_1
        franka_local_grasp_pose_1.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis_1))

        self.franka_local_grasp_pos_1 = to_torch([franka_local_grasp_pose_1.p.x, franka_local_grasp_pose_1.p.y,
                                                  franka_local_grasp_pose_1.p.z], device=self.device).repeat(
            (self.num_envs, 1))
        self.franka_local_grasp_rot_1 = to_torch([franka_local_grasp_pose_1.r.x, franka_local_grasp_pose_1.r.y,
                                                  franka_local_grasp_pose_1.r.z, franka_local_grasp_pose_1.r.w],
                                                 device=self.device).repeat((self.num_envs, 1))

        # get the cup grasp pose (should add 10mm in y axis from origin)
        cup_local_grasp_pose = gymapi.Transform()

        cup_local_grasp_pose.p.x = 0
        cup_local_grasp_pose.p.y = 0.794  # half of the cup height
        cup_local_grasp_pose.p.z = -0.29
        cup_local_grasp_pose.r = gymapi.Quat(0.0, -0.287, 0.0, 0.95793058)

        self.cup_local_grasp_pos = to_torch([cup_local_grasp_pose.p.x, cup_local_grasp_pose.p.y,
                                             cup_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        # print(self.cup_local_grasp_pos)
        self.cup_local_grasp_rot = to_torch([cup_local_grasp_pose.r.x, cup_local_grasp_pose.r.y,
                                             cup_local_grasp_pose.r.z, cup_local_grasp_pose.r.w],
                                            device=self.device).repeat((self.num_envs, 1))

        spoon_local_grasp_pose = gymapi.Transform()
        spoon_local_grasp_pose.p.x = 0
        spoon_local_grasp_pose.p.y = 0.867
        spoon_local_grasp_pose.p.z = 0.29
        spoon_local_grasp_pose.r = gymapi.Quat(0.0, 0.0, 0.6, 0.707)

        self.spoon_local_grasp_pos = to_torch([spoon_local_grasp_pose.p.x, spoon_local_grasp_pose.p.y,
                                               spoon_local_grasp_pose.p.z], device=self.device).repeat(
            (self.num_envs, 1))
        self.spoon_local_grasp_rot = to_torch([spoon_local_grasp_pose.r.x, spoon_local_grasp_pose.r.y,
                                               spoon_local_grasp_pose.r.z, spoon_local_grasp_pose.r.w],
                                              device=self.device).repeat((self.num_envs, 1))

        # set axis
        # need look details
        self.gripper_forward_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs, 1))
        self.cup_inward_axis = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))

        # print('self.gripper_forward_axis: {}, self.gripper_up_axis: {}'.format(self.gripper_forward_axis,
        #                                                                         self.gripper_up_axis))
        self.cup_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))

        self.gripper_forward_axis_1 = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.spoon_inward_axis = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis_1 = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.spoon_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.spoon_z_axis = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        if turn_spoon:
            self.spoon_inward_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs, 1))  # -z
            self.spoon_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))  # +y
        # print('self.spoon_inward_axis: {}, self.gripper_up_axis: {}'.format(self.spoon_inward_axis,
        #                                                                         self.spoon_up_axis))
        self.table_rot = to_torch([0, 0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        # params for calculation
        self.franka_grasp_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_grasp_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_grasp_rot[..., -1] = franka_local_grasp_pose.r.w  # xyzw

        self.franka_grasp_pos_1 = torch.zeros_like(self.franka_local_grasp_pos_1)
        self.franka_grasp_rot_1 = torch.zeros_like(self.franka_local_grasp_rot_1)
        self.franka_grasp_rot_1[..., -1] = franka_local_grasp_pose_1.r.w  # xyzw

        self.cup_grasp_pos = torch.zeros_like(self.cup_local_grasp_pos)
        self.cup_grasp_rot = torch.zeros_like(self.cup_local_grasp_rot)
        self.cup_grasp_rot[..., -1] = cup_local_grasp_pose.r.w

        self.spoon_grasp_pos = torch.zeros_like(self.spoon_local_grasp_pos)
        self.spoon_grasp_rot = torch.zeros_like(self.spoon_local_grasp_rot)
        self.spoon_grasp_rot[..., -1] = spoon_local_grasp_pose.r.w

        self.franka_lfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_rfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_lfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_rfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)

        self.franka_lfinger_pos_1 = torch.zeros_like(self.franka_local_grasp_pos_1)
        self.franka_rfinger_pos_1 = torch.zeros_like(self.franka_local_grasp_pos_1)
        self.franka_lfinger_rot_1 = torch.zeros_like(self.franka_local_grasp_rot_1)
        self.franka_rfinger_rot_1 = torch.zeros_like(self.franka_local_grasp_rot_1)
        ##########################################################################3
        # important

    ###########################################################
    def take_picture(self):

        count = 0
        #       for test (take one picture and save for environment 1 in a simulation )
        '''
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "../graphics_images/rgb_env%d_frame%d.png" % (count, self.index_of_simulation))
        self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.camera_handles[count][0], gymapi.IMAGE_COLOR,
                                            asset_root)
        '''

        #     save picture frame(RGBA) collection
        '''
         for envs in self.envs:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "../graphics_images/rgb_env%d_frame%d.png" % (count, self.index_of_simulation))
            # rgb_filename = "graphics_images/rgb_env%d_frame%d.png" % (count,self.index_of_simulation)
            # self.Image_rgba = self.gym.get_camera_image_gpu_tensor(self.sim, envs, self.camera_handles,
            #                                                        gymapi.IMAGE_COLOR)
            self.gym.write_camera_image_to_file(self.sim, envs, self.camera_handles[count][0], gymapi.IMAGE_COLOR,
                                                asset_root)
            print("successfully save pic graphics_images/rgb_env%d_frame%d.png" % (count, self.index_of_simulation))
            # self.torch_camera_tensor = gymtorch.wrap_tensor(self.Image_rgba)
            count += 1
        '''

        #     for picture frame(RGB-D) collection
        import torchvision
        import torchvision.transforms as transforms

        for i in range(self.num_Envs):
            camera_np = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[count][0],
                                                  gymapi.IMAGE_COLOR)
            camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[count][0],
                                                                 gymapi.IMAGE_COLOR)
            camera_tensor_grey = torchvision.transforms.functional.rgb_to_grayscale(camera_tensor, 3)
            self.norm_depth_image[i].append(camera_tensor)
            self.norm_depth_image.append([])

            count += 1

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.UsingIK:
            self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        # get pic's tensor
        self.gym.start_access_image_tensors(self.sim)
        # self.take_picture()
        self.gym.end_access_image_tensors(self.sim)

        self.index_of_simulation += 1
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]

        hand_pos_1 = self.rigid_body_states[:, self.hand_handle_1][:, 0:3]
        hand_rot_1 = self.rigid_body_states[:, self.hand_handle_1][:, 3:7]

        cup_pos = self.rigid_body_states[:, self.cup_handle][:, 0:3]
        cup_rot = self.rigid_body_states[:, self.cup_handle][:, 3:7]

        spoon_pos = self.rigid_body_states[:, self.spoon_handle][:, 0:3]
        spoon_rot = self.rigid_body_states[:, self.spoon_handle][:, 3:7]

        # franka with cup and franka1 with spoon
        self.franka_grasp_rot[:], self.franka_grasp_pos[:], self.spoon_grasp_rot[:], self.spoon_grasp_pos[:], \
        self.franka_grasp_rot_1[:], self.franka_grasp_pos_1[:], self.cup_grasp_rot[:], self.cup_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.franka_local_grasp_rot, self.franka_local_grasp_pos,
                                     cup_rot, cup_pos, self.cup_local_grasp_rot, self.cup_local_grasp_pos,
                                     hand_rot_1, hand_pos_1, self.franka_local_grasp_rot_1,
                                     self.franka_local_grasp_pos_1,
                                     spoon_rot, spoon_pos, self.spoon_local_grasp_rot, self.spoon_local_grasp_pos
                                     )

        self.franka_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3].clone()
        self.franka_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3].clone()
        self.franka_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.franka_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        v = torch.zeros_like(self.franka_lfinger_pos)
        v[:, 2] = 0.04
        self.franka_lfinger_pos += quat_rotate(self.franka_grasp_rot, v)
        self.franka_rfinger_pos += quat_rotate(self.franka_grasp_rot, v)

        self.franka_lfinger_pos_1 = self.rigid_body_states[:, self.lfinger_handle_1][:, 0:3].clone()
        self.franka_rfinger_pos_1 = self.rigid_body_states[:, self.rfinger_handle_1][:, 0:3].clone()
        self.franka_lfinger_rot_1 = self.rigid_body_states[:, self.lfinger_handle_1][:, 3:7]
        self.franka_rfinger_rot_1 = self.rigid_body_states[:, self.rfinger_handle_1][:, 3:7]

        self.franka_lfinger_pos_1 += quat_rotate(self.franka_grasp_rot_1, v)
        self.franka_rfinger_pos_1 += quat_rotate(self.franka_grasp_rot_1, v)

        dof_pos_scaled = (2.0 * (self.franka_dof_pos - self.franka_dof_lower_limits)
                          / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)
        dof_pos_scaled_1 = (2.0 * (self.franka_dof_pos_1 - self.franka_dof_lower_limits)
                            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)
        to_target = self.spoon_grasp_pos - self.franka_grasp_pos
        to_target_1 = self.cup_grasp_pos - self.franka_grasp_pos_1
        # 9 9 9 9 3 3 7 7 9 9
        # self.obs_buf = torch.cat((dof_pos_scaled, dof_pos_scaled_1,
        #                           self.franka_dof_vel * self.dof_vel_scale, to_target,
        #                           self.franka_dof_vel_1 * self.dof_vel_scale, to_target_1,
        #                           spoon_pos, spoon_rot, cup_pos, cup_rot,
        #                           self.franka_dof_pos, self.franka_dof_pos_1), dim=-1)

        self.obs_buf_left = torch.cat((dof_pos_scaled,
                                       self.franka_dof_vel * self.dof_vel_scale, to_target,
                                       spoon_pos, spoon_rot, self.franka_dof_pos), dim=-1)
        self.obs_buf_right = torch.cat((dof_pos_scaled_1,
                                        self.franka_dof_vel_1 * self.dof_vel_scale, to_target_1,
                                        cup_pos, cup_rot, self.franka_dof_pos_1), dim=-1)
        self.obs_buf = torch.cat((self.obs_buf_left, self.obs_buf_right), dim=1)

        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.reset_buf_spoon[:], self.reset_buf_cup[:], \
        self.reward_dict, self.gripped, self.gripped_1, self.left_reward_stage1[:], self.right_reward_stage1[
                                                                                    :] = compute_franka_reward(
            self.reset_buf, self.reset_buf_spoon, self.reset_buf_cup, self.progress_buf, self.progress_buf_spoon,
            self.progress_buf_cup, self.actions,
            self.franka_grasp_pos, self.cup_grasp_pos, self.franka_grasp_rot,
            self.franka_grasp_pos_1, self.spoon_grasp_pos, self.franka_grasp_rot_1,
            self.cup_grasp_rot, self.spoon_grasp_rot, self.table_rot, self.cup_positions, self.spoon_positions,
            self.cup_orientations, self.spoon_orientations, self.spoon_linvels, self.cup_linvels,
            self.cup_inward_axis, self.cup_up_axis, self.franka_lfinger_pos, self.franka_rfinger_pos,
            self.spoon_inward_axis, self.spoon_up_axis, self.franka_lfinger_pos_1, self.franka_rfinger_pos_1,
            self.gripper_forward_axis, self.gripper_up_axis,
            self.gripper_forward_axis_1, self.gripper_up_axis_1, self.contact_forces,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale,
            self.lift_reward_scale, self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset,
            self.max_episode_length, self.stage2_3_scale, self.stage2begin, self.spoon_z_axis)

        # self.reset_num1 = torch.cat((self.contact_forces[:, 0:5, :], self.contact_forces[:, 6:7, :]), 1)
        # self.reset_num2 = torch.cat((self.contact_forces[:, 10:15, :], self.contact_forces[:, 16:17, :]), 1)
        # self.reset_num = torch.cat((self.reset_num1, self.reset_num2), dim=-1)
        # self.reset_numm = torch.cat((self.contact_forces[:, 0:7, :], self.contact_forces[:, 10:17, :]), 1)

    def orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def compute_dpose(self, goal_pos, goal_rot, current_pos, current_rot):
        pos_err = goal_pos - current_pos
        ori_err = self.orientation_error(goal_rot, current_rot)
        dpose = torch.cat([pos_err, ori_err], -1).unsqueeze(-1)
        return dpose

    def reset_idx_replay_buffer(self, env_ids):

        rand_idx = 300
        rand_franka_cup_pos = self.data_actions_right[rand_idx, :]
        rand_franka_spoon_pos = self.data_actions_left[rand_idx, :]
        rand_cup_pos = self.data_obs_right[rand_idx, -16:-9]
        rand_spoon_pos = self.data_obs_left[rand_idx, -16:-9]
        pos_spoon = to_torch(rand_franka_spoon_pos, device=self.device)
        # print("pos is ", pos)
        # reset franka with "pos"
        self.franka_dof_pos[env_ids, :] = pos_spoon
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.curi_dof_targets[env_ids, 3 + self.num_franka_dofs:3 + 2 * self.num_franka_dofs] = pos_spoon

        # reset franka1
        pos_1 = to_torch(rand_franka_cup_pos, device=self.device)
        self.franka_dof_pos_1[env_ids, :] = pos_1
        self.franka_dof_vel_1[env_ids, :] = torch.zeros_like(self.franka_dof_vel_1[env_ids])
        self.curi_dof_targets[env_ids, 3: 3 + self.num_franka_dofs] = pos_1

        # # reset cup
        self.cup_positions[env_ids] = to_torch(rand_cup_pos[0:3], device=self.device)
        self.cup_orientations[env_ids] = to_torch(rand_cup_pos[3:7], device=self.device)
        self.cup_angvels[env_ids] = 0.0
        self.cup_linvels[env_ids] = 0.0
        # reset spoon
        self.spoon_positions[env_ids] = to_torch(rand_spoon_pos[0:3], device=self.device)
        self.spoon_orientations[env_ids] = to_torch(rand_spoon_pos[3:7], device=self.device)
        self.spoon_angvels[env_ids] = 0.0
        self.spoon_linvels[env_ids] = 0.0

        # reset shelf
        # self.shelf_positions[env_ids, 0] = -0.3
        # self.shelf_positions[env_ids, 1] = 0.4
        # self.shelf_positions[env_ids, 2] = 0.29
        # self.shelf_orientations[env_ids, 0] = -0.707107
        # self.shelf_orientations[env_ids, 1:3] = 0.0
        # self.shelf_orientations[env_ids, 3] = 0.707107
        # self.shelf_angvels[env_ids] = 0.0
        # self.shelf_linvels[env_ids] = 0.0

        # reset table
        # self.table_positions[env_ids, 0] = 0.0
        # self.table_positions[env_ids, 1] = 0.0
        # self.table_positions[env_ids, 2] = 0.0
        # self.table_orientations[env_ids, 0] = 0.0
        # self.table_orientations[env_ids, 1:3] = 0.0
        # self.table_orientations[env_ids, 3] = 1.0
        # self.table_angvels[env_ids] = 0.0
        # self.table_linvels[env_ids] = 0.0

        # reset root state for spoon and cup in selected envs
        actor_indices = self.global_indices[env_ids, 1:3].flatten()

        actor_indices_32 = actor_indices.to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_tensor),
                                                     gymtorch.unwrap_tensor(actor_indices_32), len(actor_indices_32))

        multi_env_ids = self.global_indices[env_ids, :1].flatten()
        multi_env_ids_int32 = multi_env_ids.to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.curi_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0

        self.reset_buf[env_ids] = 0

    def reset_idx(self, env_ids):
        pos = torch.zeros_like(self.franka_dof_state_1[..., 0]).squeeze(
            -1)  # only need [...,0]->position, 1 for velocity
        pos_1 = torch.zeros_like(self.franka_dof_state[..., 0]).squeeze(-1)
        # define rand trans by adding gaussian noise
        cup_xtrans = tensor_clamp(0.05 * torch.randn((len(env_ids)), device=self.device),
                                  to_torch([-0.05], device=self.device), to_torch([0.05], device=self.device))
        cup_ztrans = tensor_clamp(0.05 * torch.randn((len(env_ids)), device=self.device),
                                  to_torch([-0.05], device=self.device), to_torch([0.05], device=self.device))
        spoon_xtrans = tensor_clamp(0.05 * torch.randn((len(env_ids)), device=self.device),
                                    to_torch([-0.05], device=self.device), to_torch([0.05], device=self.device))
        spoon_ztrans = tensor_clamp(0.05 * torch.randn((len(env_ids)), device=self.device),
                                    to_torch([-0.05], device=self.device), to_torch([0.05], device=self.device))
        # reset cup
        if self.LocationNoise:
            self.cup_positions[env_ids, 0] = 0 + cup_xtrans
            self.cup_positions[env_ids, 2] = -0.29 + cup_ztrans
            self.box_positions[env_ids, 0] = 0 + cup_xtrans
            self.box_positions[env_ids, 2] = -0.29 + cup_ztrans
            self.spoon_positions[env_ids, 0] = 0 + spoon_xtrans
            self.spoon_positions[env_ids, 2] = 0.29 + spoon_ztrans
            self.shelf_positions[env_ids, 0] = 0 + spoon_xtrans
            self.shelf_positions[env_ids, 2] = 0.29 + spoon_ztrans
        else:
            self.cup_positions[env_ids, 0] = 0
            self.cup_positions[env_ids, 2] = -0.29
            self.box_positions[env_ids, 0] = 0
            self.box_positions[env_ids, 2] = -0.29
            self.spoon_positions[env_ids, 0] = 0
            self.spoon_positions[env_ids, 2] = 0.29
            self.shelf_positions[env_ids, 0] = 0
            self.shelf_positions[env_ids, 2] = 0.29
        self.cup_positions[env_ids, 1] = 0.792

        self.cup_orientations[env_ids, 0:3] = 0.0
        self.cup_orientations[env_ids, 1] = -0.287
        self.cup_orientations[env_ids, 3] = 0.95793058
        self.cup_linvels[env_ids] = 0.0
        self.cup_angvels[env_ids] = 0.0

        self.box_positions[env_ids, 1] = 0.770

        self.box_orientations[env_ids, 0:4] = to_torch([0.0, 0.0, 0.0, 1.0], device=self.device)
        # reset spoon

        self.spoon_positions[env_ids, 1] = 0.95
        self.spoon_orientations[env_ids, 0] = 0
        self.spoon_orientations[env_ids, 1] = 0.0
        self.spoon_orientations[env_ids, 2] = 0.707
        self.spoon_orientations[env_ids, 3] = 0.707
        self.spoon_angvels[env_ids] = 0.0
        self.spoon_linvels[env_ids] = 0.0
        self.shelf_positions[env_ids, 1] = 0.75
        self.shelf_orientations[env_ids, 0:4] = to_torch([0.0, 0.0, 0.0, 1.0], device=self.device)
        spoon_tail_pos = quat_rotate_inverse(self.spoon_orientations[env_ids, :],
                                             self.spoon_positions[env_ids, :]) + 0.5 * torch.tensor(
            [0.21, 0.0, 0.0], device=self.device)
        spoon_tail_pos = quat_rotate(self.spoon_orientations[env_ids, :], spoon_tail_pos[env_ids, :])
        # reset franka
        if self.UsingIK:
            hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
            hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
            hand_pos_1 = self.rigid_body_states[:, self.hand_handle_1][:, 0:3]
            hand_rot_1 = self.rigid_body_states[:, self.hand_handle_1][:, 3:7]
            dpose_spoon = self.compute_dpose(spoon_tail_pos, self.spoon_orientations,
                                             hand_pos, hand_rot)
            dpose_cup = self.compute_dpose(self.cup_positions, self.cup_orientations,
                                           hand_pos_1, hand_rot_1)
            left_action = self.franka_dof_pos.squeeze(-1)[:, :7] + self.control_ik(dpose_spoon, self.j_eef_left)
            right_action = self.franka_dof_pos_1.squeeze(-1)[:, :7] + self.control_ik(dpose_cup, self.j_eef_right)
            pos[:, :7] = left_action
            pos[:, 7:9] = to_torch([0.0046, 0.0046], device=self.device)
            pos_1[:, :7] = right_action
            pos_1[:, 7:9] = to_torch([0.0244, 0.0244], device=self.device)
        else:
            if self.LocationNoise:
                pos = tensor_clamp(
                    self.franka_default_dof_pos.unsqueeze(0) + 0.1 * (
                            torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
                    self.franka_dof_lower_limits, self.franka_dof_upper_limits)
                # print("pos is ", pos)
                # reset franka1
                pos_1 = tensor_clamp(
                    self.franka_default_dof_pos_1.unsqueeze(0) + 0.1 * (
                            torch.rand((len(env_ids), self.num_franka_dofs_1), device=self.device) - 0.5),
                    self.franka_dof_lower_limits, self.franka_dof_upper_limits)
            else:
                pos = tensor_clamp(
                    self.franka_default_dof_pos.unsqueeze(0),
                    self.franka_dof_lower_limits, self.franka_dof_upper_limits)
                pos_1 = tensor_clamp(
                    self.franka_default_dof_pos_1.unsqueeze(0),
                    self.franka_dof_lower_limits, self.franka_dof_upper_limits)
            self.franka_dof_pos[env_ids, :] = pos
            self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
            self.curi_dof_targets[env_ids, 3 + self.num_franka_dofs:3 + 2 * self.num_franka_dofs] = pos
            self.franka_dof_pos_1[env_ids, :] = pos_1
            self.franka_dof_vel_1[env_ids, :] = torch.zeros_like(self.franka_dof_vel_1[env_ids])
            self.curi_dof_targets[env_ids, 3:3 + self.num_franka_dofs] = pos_1
        # reset franka with "pos"
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.curi_dof_targets[env_ids, 3 + self.num_franka_dofs:3 + 2 * self.num_franka_dofs] = pos
        self.franka_dof_pos_1[env_ids, :] = pos_1
        self.franka_dof_vel_1[env_ids, :] = torch.zeros_like(self.franka_dof_vel_1[env_ids])
        self.curi_dof_targets[env_ids, 3:3 + self.num_franka_dofs] = pos_1
        # reset root state for spoon and cup in selected envs
        actor_indices = self.global_indices[env_ids, 1:5].flatten()

        actor_indices_32 = actor_indices.to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_tensor),
                                                     gymtorch.unwrap_tensor(actor_indices_32), len(actor_indices_32))

        multi_env_ids = self.global_indices[env_ids, :1].flatten()
        multi_env_ids_int32 = multi_env_ids.to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.curi_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0

        self.reset_buf[env_ids] = 0

    def reset_idx_spoon(self, env_ids):
        # reset franka
        # self.root_states[env_ids] = self.saved_root_tensor[env_ids]
        if self.ReadExpertData:
            # rand_idx = random.randrange(0, dataset1.shape[0], 1)
            rand_idx = 300
            rand_franka_cup_pos = self.data_actions_right[rand_idx, :]
            rand_franka_spoon_pos = self.data_actions_left[rand_idx, :]
            rand_cup_pos = self.data_obs_right[rand_idx, -16:-9]
            rand_spoon_pos = self.data_obs_left[rand_idx, -16:-9]
            pos = to_torch(rand_franka_spoon_pos, device=self.device)
            # print("pos is ", pos)
            # reset franka with "pos"
            self.franka_dof_pos[env_ids, :] = pos
            self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
            self.curi_dof_targets[env_ids, 3 + self.num_franka_dofs:3 + 2 * self.num_franka_dofs] = pos

            # reset spoon
            self.spoon_positions[env_ids] = to_torch(rand_spoon_pos[0:3], device=self.device)
            self.spoon_orientations[env_ids] = to_torch(rand_spoon_pos[3:7], device=self.device)
            self.spoon_angvels[env_ids] = 0.0
            self.spoon_linvels[env_ids] = 0.0

        else:
            # reset spoon
            spoon_xtrans = tensor_clamp(0.05 * torch.randn((len(env_ids)), device=self.device),
                                        to_torch([-0.05], device=self.device), to_torch([0.05], device=self.device))
            spoon_ztrans = tensor_clamp(0.05 * torch.randn((len(env_ids)), device=self.device),
                                        to_torch([-0.05], device=self.device), to_torch([0.05], device=self.device))
            # reset spoon
            self.spoon_positions[env_ids, 0] = 0 + spoon_xtrans
            self.spoon_positions[env_ids, 1] = 0.95
            self.spoon_positions[env_ids, 2] = 0.29 + spoon_ztrans
            self.spoon_orientations[env_ids, 0] = 0
            self.spoon_orientations[env_ids, 1] = 0.0
            self.spoon_orientations[env_ids, 2] = 0.707
            self.spoon_orientations[env_ids, 3] = 0.707
            self.spoon_angvels[env_ids] = 0.0
            self.spoon_linvels[env_ids] = 0.0
            self.shelf_positions[env_ids, 0] = 0 + spoon_xtrans
            self.shelf_positions[env_ids, 1] = 0.75
            self.shelf_positions[env_ids, 2] = 0.29 + spoon_ztrans
            self.shelf_orientations[env_ids, 0:4] = to_torch([0.0, 0.0, 0.0, 1.0], device=self.device)
            pos = tensor_clamp(
                self.franka_default_dof_pos.unsqueeze(0) + 0.1 * (
                        torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
                self.franka_dof_lower_limits, self.franka_dof_upper_limits)
            # print("pos is ", pos)
            # reset franka with "pos"
            self.franka_dof_pos[env_ids, :] = pos
            self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
            self.curi_dof_targets_spoon[env_ids, 3 + self.num_franka_dofs:3 + 2 * self.num_franka_dofs] = pos
            self.curi_dof_targets[env_ids, 3 + self.num_franka_dofs:3 + 2 * self.num_franka_dofs] = pos
            self.curi_dof_targets_spoon[env_ids,
            3: 3 + self.num_franka_dofs] = self.franka_dof_pos_1[
                                           env_ids, :]
        # reset root state for spoon and cup in selected envs
        actor_indices = self.global_indices[env_ids, 1:5].flatten()

        actor_indices_32 = actor_indices.to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_tensor),
                                                     gymtorch.unwrap_tensor(actor_indices_32), len(actor_indices_32))

        multi_env_ids = self.global_indices[env_ids, :1].flatten()
        multi_env_ids_int32 = multi_env_ids.to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.curi_dof_targets_spoon),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf_spoon[env_ids] = 0

        self.reset_buf_spoon[env_ids] = 0

    def reset_idx_cup(self, env_ids):
        # reset franka
        # self.root_states[env_ids] = self.saved_root_tensor[env_ids]
        if self.stage2begin:
            rand_idx = 300
            rand_franka_cup_pos = self.data_actions_right[rand_idx, :]
            rand_franka_spoon_pos = self.data_actions_left[rand_idx, :]
            rand_cup_pos = self.data_obs_right[rand_idx, -16:-9]
            rand_spoon_pos = self.data_obs_left[rand_idx, -16:-9]

            # reset franka1
            pos_1 = to_torch(rand_franka_cup_pos, device=self.device)
            self.franka_dof_pos_1[env_ids, :] = pos_1
            self.franka_dof_vel_1[env_ids, :] = torch.zeros_like(self.franka_dof_vel_1[env_ids])
            self.curi_dof_targets[env_ids, 3:3 + self.num_franka_dofs] = pos_1

            # # reset cup
            self.cup_positions[env_ids] = to_torch(rand_cup_pos[0:3], device=self.device)
            self.cup_orientations[env_ids] = to_torch(rand_cup_pos[3:7], device=self.device)
            self.cup_angvels[env_ids] = 0.0
            self.cup_linvels[env_ids] = 0.0

        # reset franka1
        else:
            pos_1 = tensor_clamp(
                self.franka_default_dof_pos_1.unsqueeze(0) + 0.1 * (
                        torch.rand((len(env_ids), self.num_franka_dofs_1), device=self.device) - 0.5),
                self.franka_dof_lower_limits, self.franka_dof_upper_limits)
            self.franka_dof_pos_1[env_ids, :] = pos_1
            self.franka_dof_vel_1[env_ids, :] = torch.zeros_like(self.franka_dof_vel_1[env_ids])

            self.curi_dof_targets_cup[env_ids,
            3 + self.num_franka_dofs:3 + 2 * self.num_franka_dofs] = self.franka_dof_pos[env_ids, :]
            self.curi_dof_targets_cup[env_ids, 3:3 + self.num_franka_dofs] = pos_1
            self.curi_dof_targets[env_ids, 3:3 + self.num_franka_dofs] = pos_1
            cup_xtrans = tensor_clamp(0.05 * torch.randn((len(env_ids)), device=self.device),
                                      to_torch([-0.05], device=self.device), to_torch([0.05], device=self.device))
            cup_ztrans = tensor_clamp(0.05 * torch.randn((len(env_ids)), device=self.device),
                                      to_torch([-0.05], device=self.device), to_torch([0.05], device=self.device))
            # reset cup
            self.cup_positions[env_ids, 0] = 0 + cup_xtrans
            self.cup_positions[env_ids, 1] = 0.792
            self.cup_positions[env_ids, 2] = -0.29 + cup_ztrans
            self.cup_orientations[env_ids, 0:3] = 0.0
            self.cup_orientations[env_ids, 1] = -0.287
            self.cup_orientations[env_ids, 3] = 0.95793058
            self.cup_linvels[env_ids] = 0.0
            self.cup_angvels[env_ids] = 0.0
            self.box_positions[env_ids, 0] = 0 + cup_xtrans
            self.box_positions[env_ids, 1] = 0.770
            self.box_positions[env_ids, 2] = -0.29 + cup_ztrans
            self.box_orientations[env_ids, 0:4] = to_torch([0.0, 0.0, 0.0, 1.0], device=self.device)

        # reset root state for spoon and cup in selected envs
        actor_indices = self.global_indices[env_ids, 1:5].flatten()

        actor_indices_32 = actor_indices.to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_tensor),
                                                     gymtorch.unwrap_tensor(actor_indices_32), len(actor_indices_32))

        multi_env_ids = self.global_indices[env_ids, :1].flatten()
        multi_env_ids_int32 = multi_env_ids.to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.curi_dof_targets_cup),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf_cup[env_ids] = 0

        self.reset_buf_cup[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)  # (num_envs, 18)

        # compute franka next target
        # relative control
        targets = self.curi_dof_targets[:,
                  3 + self.num_franka_dofs:3 + 2 * self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions[
                                                                                                                    :,
                                                                                                                    0:9] * self.action_scale
        self.curi_dof_targets[:, 3 + self.num_franka_dofs:3 + 2 * self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        targets_1 = self.curi_dof_targets[:,
                    3:3 + self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt \
                    * self.actions[:, 9:18] * self.action_scale
        self.curi_dof_targets[:, 3:3 + self.num_franka_dofs] = tensor_clamp(
            targets_1, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # grip_act_spoon=torch.where(self.gripped==1,torch.Tensor([[0.004, 0.004]] * self.num_envs).to(self.device), torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device))
        # self.curi_dof_targets[:, -1] = torch.where(self.gripped == 1, 0.0046, 0.04)
        # self.curi_dof_targets[:, -2] = torch.where(self.gripped == 1, 0.0046, 0.04)
        ## for offlinedata collection in test.py
        self.curi_dof_targets[:, -1] = torch.where(self.curi_dof_targets[:, -1] < 0.005, 0.0035,
                                                   self.curi_dof_targets[:, -1].to(torch.double))
        self.curi_dof_targets[:, -2] = torch.where(self.curi_dof_targets[:, -2] < 0.005, 0.0035,
                                                   self.curi_dof_targets[:, -2].to(torch.double))
        self.curi_dof_targets[:, 10] = torch.where(self.gripped_1 == 1, 0.024, 0.04)
        self.curi_dof_targets[:, 11] = torch.where(self.gripped_1 == 1, 0.024, 0.04)
        # give to gym
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.curi_dof_targets))

    def post_physics_step(self):  # what do frankas do after interacting with the envs
        self.progress_buf += 1
        self.progress_buf_cup += 1
        self.progress_buf_spoon += 1
        # print("progress buffer is ",self.progress_buf)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        env_ids_spoon = self.reset_buf_spoon.nonzero(as_tuple=False).squeeze(-1)
        env_ids_cup = self.reset_buf_cup.nonzero(as_tuple=False).squeeze(-1)

        # all reset
        # if len(env_ids) > 0:
        #     if self.ResetFromReplay == True:
        #         self.reset_idx_replay_buffer(env_ids)
        #     else:
        #         self.reset_idx(env_ids)

        # reset respective
        if len(env_ids_cup) > 0:
            self.reset_idx_cup(env_ids_cup)
        if len(env_ids_spoon) > 0:
            self.reset_idx_spoon(env_ids_spoon)

        self.compute_observations()
        self.compute_reward()
        # debug viz, show axis
        if self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                px = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([1, 0, 0],
                                                                                               device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([0, 1, 0],
                                                                                               device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([0, 0, 1],
                                                                                               device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]],
                                   [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]],
                                   [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
                                   [0.1, 0.1, 0.85])

                px = (self.franka_grasp_pos_1[i] + quat_apply(self.franka_grasp_rot_1[i], to_torch([1, 0, 0],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_grasp_pos_1[i] + quat_apply(self.franka_grasp_rot_1[i], to_torch([0, 1, 0],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_grasp_pos_1[i] + quat_apply(self.franka_grasp_rot_1[i], to_torch([0, 0, 1],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_grasp_pos_1[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]],
                                   [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]],
                                   [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
                                   [0.1, 0.1, 0.85])

                px = (self.cup_grasp_pos[i] + quat_apply(self.cup_grasp_rot[i],
                                                         to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.cup_grasp_pos[i] + quat_apply(self.cup_grasp_rot[i],
                                                         to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.cup_grasp_pos[i] + quat_apply(self.cup_grasp_rot[i],
                                                         to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.cup_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                # px = ( quat_rotate(self.spoon_grasp_pos[i], quat_rotate_inverse(self.spoon_grasp_rot[i], self.spoon_grasp_pos[i]) + 0.5 * torch.tensor(
                #     [0.21, 0.0, 0.0],
                #     device=self.device)) + quat_apply(self.spoon_grasp_rot[i],
                #                                          to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = ( quat_rotate(self.spoon_grasp_pos[i], quat_rotate_inverse(self.spoon_grasp_rot[i], self.spoon_grasp_pos[i]) + 0.5 * torch.tensor(
                #     [0.21, 0.0, 0.0],
                #     device=self.device)) + quat_apply(self.spoon_grasp_rot[i],
                #                                          to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = ( quat_rotate(self.spoon_grasp_pos[i], quat_rotate_inverse(self.spoon_grasp_rot[i], self.spoon_grasp_pos[i]) + 0.5 * torch.tensor(
                #     [0.21, 0.0, 0.0],
                #     device=self.device)) + quat_apply(self.spoon_grasp_rot[i],
                #                                          to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.spoon_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([1, 0, 0],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 1, 0],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 0, 1],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([1, 0, 0],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 1, 0],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 0, 1],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_lfinger_pos_1[i] + quat_apply(self.franka_lfinger_rot_1[i], to_torch([1, 0, 0],
                                                                                                       device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_lfinger_pos_1[i] + quat_apply(self.franka_lfinger_rot_1[i], to_torch([0, 1, 0],
                                                                                                       device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_lfinger_pos_1[i] + quat_apply(self.franka_lfinger_rot_1[i], to_torch([0, 0, 1],
                                                                                                       device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_lfinger_pos_1[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_rfinger_pos_1[i] + quat_apply(self.franka_rfinger_rot_1[i], to_torch([1, 0, 0],
                                                                                                       device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_rfinger_pos_1[i] + quat_apply(self.franka_rfinger_rot_1[i], to_torch([0, 1, 0],
                                                                                                       device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_rfinger_pos_1[i] + quat_apply(self.franka_rfinger_rot_1[i], to_torch([0, 0, 1],
                                                                                                       device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_rfinger_pos_1[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

    def set_viewer(self):
        """Create the viewer."""

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
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_P, "myprint")

            # Point camera at middle env
            num_per_row = int(math.sqrt(self.num_envs))
            cam_pos = gymapi.Vec3(4, 3, 2)
            cam_target = gymapi.Vec3(-4, -3, 0)
            middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
        reset_buf, reset_buf_spoon, reset_buf_cup, progress_buf, progress_buf_spoon, progress_buf_cup, actions,
        franka_grasp_pos, cup_grasp_pos, franka_grasp_rot, franka_grasp_pos_1,
        spoon_grasp_pos, franka_grasp_rot_1, cup_grasp_rot, spoon_grasp_rot, table_rot,
        cup_positions, spoon_positions, cup_orientations, spoon_orientations,
        spoon_linvels, cup_linvels,
        cup_inward_axis, cup_up_axis, franka_lfinger_pos, franka_rfinger_pos,
        spoon_inward_axis, spoon_up_axis, franka_lfinger_pos_1, franka_rfinger_pos_1,
        gripper_forward_axis, gripper_up_axis,
        gripper_forward_axis_1, gripper_up_axis_1, contact_forces,
        num_envs: int, dist_reward_scale: float, rot_reward_scale: float, around_handle_reward_scale: float,
        lift_reward_scale: float, finger_dist_reward_scale: float, action_penalty_scale: float, distX_offset: float,
        max_episode_length: float, stage2_3_scale: float, stage2begin: bool, spoon_z_axis
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[
    str, Union[Dict[str, Tuple[Tensor, Union[Tensor, float]]], Dict[str, Tuple[Tensor, float]], Dict[
        str, Tensor]]], Tensor, Tensor, Tensor, Tensor]:
    """
    Tuple[Tensor, Tensor, Dict[str, Union[Dict[str, Tuple[Tensor, float]],
                                           Dict[str, Tensor], Dict[str, Union[Tensor, Tuple[Tensor, float]]]]]]:
    """
    tensor_device = franka_grasp_pos.device
    spoon_tail_pos = quat_rotate_inverse(spoon_orientations, spoon_positions) + 0.5 * torch.tensor([0.21, 0.0, 0.0],
                                                                                                   device=tensor_device)
    spoon_tail_pos = quat_rotate(spoon_orientations, spoon_tail_pos)
    # turn_spoon = True

    d = torch.norm(franka_grasp_pos - spoon_tail_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.06, dist_reward * 2, dist_reward)
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)  # TODO: test

    d_1 = torch.norm(franka_grasp_pos_1 - cup_positions, p=2, dim=-1)
    dist_reward_1 = 1.0 / (1.0 + d_1 ** 2)
    dist_reward_1 *= dist_reward_1
    dist_reward_1 = torch.where(d_1 <= 0.06, dist_reward_1 * 2, dist_reward_1)
    # </editor-fold>

    axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)  # franka

    axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
    axis4 = tf_vector(spoon_orientations, spoon_inward_axis)  # [1,0,0]

    axis1_1 = tf_vector(franka_grasp_rot_1, gripper_forward_axis_1)  # franka1
    axis2_1 = tf_vector(cup_orientations, cup_inward_axis)
    axis3_1 = tf_vector(franka_grasp_rot_1, gripper_up_axis_1)
    axis4_1 = tf_vector(cup_orientations, cup_up_axis)

    axistable = tf_vector(table_rot, cup_up_axis)  # add ground axis

    # compute the alignment(reward)
    # alignment of forward axis for gripper
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # franka-x with spoon-x

    '''Box and turn'''

    dot1_1 = torch.bmm(axis1_1.view(num_envs, 1, 3), axis2_1.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # franka-z with cup-x
    dot2_1 = torch.bmm(axis3_1.view(num_envs, 1, 3), axis4_1.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # franka-x with cup-y

    # reward for matching the orientation of the hand to the cup(fingers wrapped)
    rot_reward = (torch.sign(dot2) * dot2 ** 2)
    rot_reward_1 = 0.5 * (torch.sign(dot1_1) * dot1_1 ** 2 + torch.sign(dot2_1) * dot2_1 ** 2)
    # </editor-fold>

    '''Discrete'''
    spoon_size = [0.21, 0.01, 0.01]
    around_handle_reward = torch.zeros_like(rot_reward)
    around_handle_reward = torch.where(quat_rotate_inverse(spoon_orientations, franka_lfinger_pos)[:, 2] \
                                       > quat_rotate_inverse(spoon_orientations, spoon_positions)[:, 2] - (
                                                spoon_size[2]),
                                       torch.where(quat_rotate_inverse(spoon_orientations, franka_rfinger_pos)[:, 2] \
                                                   < quat_rotate_inverse(spoon_orientations, spoon_positions)[:, 2] + (
                                                          spoon_size[2]),
                                                   around_handle_reward + 0.5, around_handle_reward),
                                       around_handle_reward)
    around_handle_reward = torch.where(quat_rotate_inverse(spoon_orientations, franka_lfinger_pos)[:, 0] \
                                       > quat_rotate_inverse(spoon_orientations, spoon_positions)[:, 0],
                                       torch.where(quat_rotate_inverse(spoon_orientations, franka_rfinger_pos)[:, 0] \
                                                   < quat_rotate_inverse(spoon_orientations, spoon_positions)[:, 0] + (
                                                           0.5 * spoon_size[0]),
                                                   around_handle_reward + 0.5, around_handle_reward),
                                       around_handle_reward)
    around_handle_reward = torch.where(quat_rotate_inverse(spoon_orientations, franka_lfinger_pos)[:, 1] \
                                       <(quat_rotate_inverse(spoon_orientations, spoon_positions)[:, 1]+0.02),
                                       torch.where(quat_rotate_inverse(spoon_orientations, franka_rfinger_pos)[:, 1] \
                                                   > quat_rotate_inverse(spoon_orientations, spoon_positions)[:, 1] + (
                                                       spoon_size[1]-0.02),
                                                   around_handle_reward + 0.5, around_handle_reward),
                                       around_handle_reward)
    gripped = (around_handle_reward == 1.5)
    cup_size = [0.05, 0.1, 0.05]
    around_handle_reward_1 = torch.zeros_like(rot_reward_1)
    around_handle_reward_1 = torch.where(quat_rotate_inverse(cup_orientations, franka_lfinger_pos_1)[:, 2] \
                                         > quat_rotate_inverse(cup_orientations, cup_positions)[:, 2] - (
                                                 0.5 * cup_size[2] + 0.002),
                                         torch.where(quat_rotate_inverse(cup_orientations, franka_rfinger_pos_1)[:, 2] \
                                                     < quat_rotate_inverse(cup_orientations, cup_positions)[:, 2] + (
                                                             0.5 * cup_size[2] + 0.002),
                                                     around_handle_reward_1 + 0.5, around_handle_reward_1),
                                         around_handle_reward_1)
    around_handle_reward_1 = torch.where(quat_rotate_inverse(cup_orientations, franka_lfinger_pos_1)[:, 0] \
                                         > quat_rotate_inverse(cup_orientations, cup_positions)[:, 0] - (
                                                 0.5 * cup_size[0] + 0.002),
                                         torch.where(quat_rotate_inverse(cup_orientations, franka_rfinger_pos_1)[:, 0] \
                                                     < quat_rotate_inverse(cup_orientations, cup_positions)[:, 0] + (
                                                             0.5 * cup_size[0] + 0.002),
                                                     around_handle_reward_1 + 0.5, around_handle_reward_1),
                                         around_handle_reward_1)
    around_handle_reward_1 = torch.where(quat_rotate_inverse(cup_orientations, franka_lfinger_pos_1)[:, 1] \
                                         > quat_rotate_inverse(cup_orientations, cup_positions)[:, 1],
                                         torch.where(quat_rotate_inverse(cup_orientations, franka_rfinger_pos_1)[:, 1] \
                                                     < quat_rotate_inverse(cup_orientations, cup_positions)[:, 1] + (
                                                         cup_size[1]),
                                                     around_handle_reward_1 + 0.5, around_handle_reward_1),
                                         around_handle_reward_1)
    gripped_1 = (around_handle_reward_1 == 1.5)

    '''Discrete'''
    finger_dist_reward = torch.zeros_like(rot_reward)
    lfinger_dist = quat_rotate_inverse(spoon_orientations, franka_lfinger_pos - spoon_tail_pos)[:, 2]
    rfinger_dist = quat_rotate_inverse(spoon_orientations, franka_rfinger_pos - spoon_tail_pos)[:, 2]
    lfinger_dist = torch.where(lfinger_dist > 0, lfinger_dist, lfinger_dist + 0.002)
    rfinger_dist = torch.where(rfinger_dist < 0, rfinger_dist, rfinger_dist - 0.002)
    tmp = torch.clamp(torch.abs(lfinger_dist - rfinger_dist), 0.0105, 0.08)
    finger_dist_reward = torch.where(lfinger_dist > 0,
                                     torch.where(rfinger_dist < 0,
                                                 (0.08 - tmp),
                                                 finger_dist_reward),
                                     finger_dist_reward)
    finger_dist_reward = torch.where(lfinger_dist > 0,
                                     torch.where(rfinger_dist < 0,
                                                 torch.where(d <= 0.02,
                                                             (0.08 - tmp) * 100,
                                                             finger_dist_reward), finger_dist_reward),
                                     finger_dist_reward)
    finger_dist_reward_1 = torch.zeros_like(rot_reward_1)
    lfinger_dist_1 = quat_rotate_inverse(cup_orientations, franka_lfinger_pos_1 - cup_positions)[:, 2]
    rfinger_dist_1 = quat_rotate_inverse(cup_orientations, franka_rfinger_pos_1 - cup_positions)[:, 2]
    lfinger_dist_1 = torch.where(lfinger_dist_1 > 0, lfinger_dist_1, lfinger_dist_1 + 0.020)
    rfinger_dist_1 = torch.where(rfinger_dist_1 < 0, rfinger_dist_1, rfinger_dist_1 - 0.020)
    tmp_1 = torch.clamp(torch.abs(lfinger_dist_1 - rfinger_dist_1), 0.0505, 0.08)
    finger_dist_reward_1 = torch.where(lfinger_dist_1 > 0,
                                       torch.where(rfinger_dist_1 < 0,
                                                   (0.08 - tmp_1),
                                                   finger_dist_reward_1),
                                       finger_dist_reward_1)
    finger_dist_reward_1 = torch.where(lfinger_dist_1 > 0,
                                       torch.where(rfinger_dist_1 < 0,
                                                   torch.where(d_1 <= 0.02,
                                                               (0.08 - tmp_1) * 100,
                                                               finger_dist_reward_1), finger_dist_reward_1),
                                       finger_dist_reward_1)

    cup_fall_penalty = torch.where(cup_positions[:, 1] < 0.439, 1.0, 0.0)
    dot_cup_reverse = torch.bmm(axis4_1.view(num_envs, 1, 3), cup_up_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # cup rotation y align with ground y(=cup up axis)

    spoon_fall_penalty = torch.where(spoon_positions[:, 1] < 0.48, 1.0, 0.0)

    # regularization on the actions (summed for each environment) (more actions more penalty)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # the higher the y coordinates of objects are, the larger the rewards will be set

    init_spoon_pos = torch.tensor([0.0, 0.9, 0.29])  # TODO: need to be changed
    init_cup_pos = torch.tensor([0.0, 0.792, -0.29])
    lift_reward = torch.zeros_like(rot_reward)
    lift_dist = spoon_positions[:, 1] - init_spoon_pos[1]
    lift_reward = torch.where(lift_dist < 0, lift_dist * around_handle_reward + lift_dist, lift_reward)
    lift_reward = torch.where(lift_dist > 0,
                              (lift_dist * around_handle_reward + lift_dist + lift_dist * finger_dist_reward * 10) * 5,
                              lift_reward)

    lift_reward_1 = torch.zeros_like(rot_reward_1)
    lift_dist_1 = cup_positions[:, 1] - init_cup_pos[1]
    lift_reward_1 = torch.where(lift_dist_1 < 0, lift_dist_1 * around_handle_reward_1 + lift_dist_1, lift_reward_1)
    lift_reward_1 = torch.where(lift_dist_1 > 0, (
            lift_dist_1 * around_handle_reward_1 + lift_dist_1 + lift_dist_1 * finger_dist_reward_1 * 10) * 5,
                                lift_reward_1)  # 3
    # </editor-fold>

    # ....................stage 2 reward....................................................................
    # #delete the y column
    spoon_positions_trans = spoon_positions.t()
    cup_positions_trans = cup_positions.t()
    idx = 1
    spoon_positions_without_y = spoon_positions_trans[torch.arange(spoon_positions_trans.size(0)) != idx]
    cup_positions_without_y = cup_positions_trans[torch.arange(cup_positions_trans.size(0)) != idx]
    d_spoon_cup = torch.norm(spoon_positions_without_y.t() - cup_positions_without_y.t(), p=2, dim=-1)

    # dist_reward = 2.0 / (1.0 + d ** 2)
    dist_reward_stage2 = 1.0 / (1.0 + d_spoon_cup ** 2)
    dist_reward_stage2 *= dist_reward_stage2

    dot_stage2 = torch.bmm(axis4_1.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # spoon z with cup-y
    dot_stage2_table = torch.bmm(axistable.view(num_envs, 1, 3), axis4_1.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # spoon z with cup-y
    rot_reward_stage2 = 0.5 * (
            torch.sign(dot_stage2) * dot_stage2 ** 2 + torch.sign(dot_stage2_table) * dot_stage2_table ** 2)

    d_spoon_cup_y = torch.norm(franka_grasp_pos[:, 1] - franka_grasp_pos_1[:, 1], p=2, dim=-1)
    dist_reward_stage2_y = torch.zeros_like(dist_reward)
    # dist_reward_stage2_y = torch.where(franka_grasp_pos[:, 1] > franka_grasp_pos_1[:, 1],
    #                                    torch.where(d_spoon_cup_y > 0.4, 1.0 / (1.0 + d_spoon_cup_y ** 2),
    #                                                dist_reward_stage2_y),
    #                                    dist_reward_stage2_y)
    # dist_reward_stage2_y *= dist_reward_stage2_y

    # ....................stage 3 reward....................................................................
    preset_h = 0.09
    cup_r = 0.025

    spoon_tip_pos = quat_rotate_inverse(spoon_orientations, spoon_positions) - 0.5 * torch.tensor([0.21, 0.0, 0.0],
                                                                                                  device=tensor_device)
    spoon_tip_pos = quat_rotate(spoon_orientations, spoon_tip_pos)
    v1_s3 = quat_rotate_inverse(cup_orientations, spoon_tip_pos - cup_positions)  # relative spoon pos in cup
    prestage_s3 = [torch.gt(torch.tensor([cup_r, cup_r], device=tensor_device), v1_s3[:, [0, 2]]).all(dim=-1),
                   torch.lt(torch.tensor([-cup_r, -cup_r], device=tensor_device), v1_s3[:, [0, 2]]).all(dim=-1),
                   # x,z in cup
                   torch.logical_and(v1_s3[:, 1] - preset_h < 0, v1_s3[:, 1] > 0)]  # spoon tip in cup
    stage_s3 = torch.logical_and(v1_s3[:, 1] - 0.11 < 0, v1_s3[:, 1] > 0)
    flag_range_s3 = torch.logical_and(prestage_s3[0], prestage_s3[1])
    flag_full_s3 = torch.logical_and(flag_range_s3, prestage_s3[2])
    h_s3 = torch.abs(v1_s3[:, 1] - preset_h)
    h_reward_s3 = 2.0 / (1.0 + h_s3 ** 2) * flag_range_s3
    d_s3 = torch.norm(v1_s3[:, [0, 2]] - torch.tensor([cup_r, cup_r], device=tensor_device), dim=-1)
    d_reward_s3 = 2.0 / (1.0 + d_s3 ** 2) * flag_range_s3
    spoon_v = torch.norm(spoon_linvels, dim=-1)
    spoon_v = torch.clamp(spoon_v - 0.049, min=0)
    cup_v = torch.norm(cup_linvels, dim=-1)
    cup_v = torch.clamp(cup_v - 0.049, min=0)
    v_reward_s3 = spoon_v * flag_full_s3

    h_reward_s3 = torch.where(stage_s3, 10 * h_reward_s3, h_reward_s3)
    d_reward_s3 = torch.where(stage_s3, 10 * d_reward_s3, d_reward_s3)
    v_reward_s3 = torch.where(torch.logical_and(stage_s3, cup_v < 0.1), 1000 * v_reward_s3, v_reward_s3)

    # ................................................................................................................
    ## sum of rewards
    sf = 1  # spoon flag
    cf = 1  # cup flag
    if stage2begin:
        stage1 = 0  # stage1 flag
    else:
        stage1 = 1
    stage2 = 0  # stage2 flag
    stage3 = 0  # stage3 flag
    rewards = stage1 * (dist_reward_scale * (dist_reward * sf + dist_reward_1 * cf) \
                        + rot_reward_scale * (rot_reward * sf + rot_reward_1 * cf) \
                        + around_handle_reward_scale * (around_handle_reward * sf + around_handle_reward_1 * cf) \
                        + finger_dist_reward_scale * (finger_dist_reward * sf + finger_dist_reward_1 * cf) \
                        + 20 * (lift_reward * sf) + 20 * lift_reward_1 * cf \
                        - action_penalty_scale * action_penalty \
                        - spoon_fall_penalty)

    rewards = rewards + stage2 * (dist_reward_stage2 * dist_reward_scale * 20 \
                                  + rot_reward_stage2 * rot_reward_scale * 20 \
                                  + dist_reward_stage2_y * dist_reward_scale * 20)

    # TODO: add stage 3 reward
    fulfill_s1 = torch.logical_and(spoon_positions[:, 1] - 0.4 > 0.15,
                                   # spoon_y - table_height > x  (shelf height ignored)
                                   cup_positions[:, 1] - 0.4 > 0.15,
                                   )

    rewards = rewards + stage3 * fulfill_s1 * (h_reward_s3 * 20 \
                                               + d_reward_s3 * 20 \
                                               + v_reward_s3 * 20)

    # left and right reward
    left_reward_stage1 = stage1 * (dist_reward_scale * (dist_reward * sf) \
                                   + rot_reward_scale * (rot_reward * sf) \
                                   + around_handle_reward_scale * (around_handle_reward * sf) \
                                   + finger_dist_reward_scale * (finger_dist_reward * sf) \
                                   + 20 * (lift_reward * sf) \
                                   - action_penalty_scale * action_penalty
                                   - spoon_fall_penalty)
    right_reward_stage1 = stage1 * (dist_reward_scale * (dist_reward_1 * cf) \
                                    + rot_reward_scale * (rot_reward_1 * cf) \
                                    + around_handle_reward_scale * (around_handle_reward_1 * cf) \
                                    + finger_dist_reward_scale * (finger_dist_reward_1 * cf) \
                                    + 20 * (lift_reward_1 * cf) \
                                    - action_penalty_scale * action_penalty
                                    - cup_fall_penalty
                                    )
    if stage2begin:
        left_reward_stage2 = stage2 * (dist_reward_stage2 * dist_reward_scale * stage2_3_scale \
                                       + rot_reward_stage2 * rot_reward_scale * stage2_3_scale \
                                       + dist_reward_stage2_y * dist_reward_scale * stage2_3_scale \
                                       )
        right_reward_stage2 = stage2 * (dist_reward_stage2 * dist_reward_scale * stage2_3_scale \
                                        + rot_reward_stage2 * rot_reward_scale * stage2_3_scale \
                                        + dist_reward_stage2_y * dist_reward_scale * stage2_3_scale \
                                        + 1 * (lift_reward_1 * cf))
    else:
        left_reward_stage2 = stage2 * fulfill_s1 * (dist_reward_stage2 * dist_reward_scale * stage2_3_scale \
                                                    + rot_reward_stage2 * rot_reward_scale * stage2_3_scale \
                                                    + dist_reward_stage2_y * dist_reward_scale * stage2_3_scale)
        right_reward_stage2 = stage2 * fulfill_s1 * (dist_reward_stage2 * dist_reward_scale * stage2_3_scale \
                                                     + rot_reward_stage2 * rot_reward_scale * stage2_3_scale \
                                                     + dist_reward_stage2_y * dist_reward_scale * stage2_3_scale)
    left_reward_stage3 = stage3 * fulfill_s1 * (h_reward_s3 * stage2_3_scale \
                                                + d_reward_s3 * stage2_3_scale \
                                                + v_reward_s3 * stage2_3_scale)
    right_reward_stage3 = stage3 * fulfill_s1 * (h_reward_s3 * stage2_3_scale \
                                                 + d_reward_s3 * stage2_3_scale \
                                                 + v_reward_s3 * stage2_3_scale)
    left_reward = left_reward_stage1 + left_reward_stage2 + left_reward_stage3
    right_reward = right_reward_stage1 + right_reward_stage2 + right_reward_stage3
    # test args
    '''taken up too high'''
    reset_buf = torch.where(spoon_positions[:, 1] > 1.9, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(cup_positions[:, 1] > 1.1, torch.ones_like(reset_buf), reset_buf)  #
    '''fall'''
    reset_buf = torch.where(cup_positions[:, 1] < 0.75, torch.ones_like(reset_buf),
                            reset_buf)  # cup fall to table or ground
    reset_buf = torch.where(torch.acos(dot_cup_reverse) * 180 / torch.pi > 90, torch.ones_like(reset_buf),
                            reset_buf)  # cup fall direction
    reset_buf = torch.where(spoon_positions[:, 1] < 0.8, torch.ones_like(reset_buf),
                            reset_buf)  # spoon fall to table or ground

    # # cup fall to table
    # reset_buf = torch.where(reset_numm > 400, torch.ones_like(reset_buf), reset_buf)
    # reset when max_episode_length
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    # reset_buf_spoon
    reset_buf_spoon = torch.where(spoon_positions[:, 1] > 1.9, torch.ones_like(reset_buf_spoon), reset_buf_spoon)
    reset_buf_spoon = torch.where(spoon_positions[:, 1] < 0.8, torch.ones_like(reset_buf_spoon),
                                  reset_buf_spoon)
    reset_buf_spoon = torch.where(progress_buf_spoon >= max_episode_length - 1, torch.ones_like(reset_buf_spoon),
                                  reset_buf_spoon)

    # reset_buf_cup
    reset_buf_cup = torch.where(cup_positions[:, 1] > 1.1, torch.ones_like(reset_buf_cup), reset_buf_cup)  #
    '''fall'''
    reset_buf_cup = torch.where(cup_positions[:, 1] < 0.75, torch.ones_like(reset_buf_cup),
                                reset_buf_cup)  # cup fall to table or ground
    reset_buf_cup = torch.where(torch.acos(dot_cup_reverse) * 180 / torch.pi > 90, torch.ones_like(reset_buf_cup),
                                reset_buf_cup)  # cup fall direction
    reset_buf_cup = torch.where(progress_buf_cup >= max_episode_length - 1, torch.ones_like(reset_buf_cup),
                                reset_buf_cup)
    # </editor-fold>

    # list rewards details for test
    # assert True
    reward_franka_0 = {
        'distance': (dist_reward, dist_reward_scale),
        'rotation': (rot_reward, rot_reward_scale),
        'around_hand': (around_handle_reward, around_handle_reward_scale),
        'finger_distance': (finger_dist_reward, finger_dist_reward_scale),
        'l and r distance': (lfinger_dist, rfinger_dist)
    }
    reward_franka_1 = {
        'distance': (dist_reward_1, dist_reward_scale),
        'rotation': (rot_reward_1, rot_reward_scale),
        'around_hand': (around_handle_reward_1, around_handle_reward_scale),
        'finger_distance': (finger_dist_reward_1, finger_dist_reward_scale),
    }
    rewards_other = {
        'dist_reward_stage2': (dist_reward_stage2, dist_reward_scale),
        'rot_reward_stage2': (rot_reward_stage2, rot_reward_scale),
        'dist_reward_stage2_y': (dist_reward_stage2_y, dist_reward_scale),
        'l and r distance': (lfinger_dist, rfinger_dist)
    }
    rewards_bonus = {

        'hrew': h_reward_s3,
        'drew': d_reward_s3,
        'vrew': v_reward_s3,
    }

    # if False:
    #     print(prestage_s3)
    # output dict
    rewards_dict = {
        'Franka_0': reward_franka_0,
        'Franka_1': reward_franka_1,
        'others': rewards_other,
        'bonus': rewards_bonus,
    }

    return rewards, reset_buf, reset_buf_spoon, reset_buf_cup, rewards_dict, gripped, gripped_1, left_reward, right_reward


# compute
@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos,
                             cup_rot, cup_pos, cup_local_grasp_rot, cup_local_grasp_pos, hand_rot_1, hand_pos_1,
                             franka_local_grasp_rot_1, franka_local_grasp_pos_1,
                             spoon_rot, spoon_pos, spoon_local_grasp_rot, spoon_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor]

    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos)
    global_spoon_rot, global_spoon_pos = tf_combine(
        spoon_rot, spoon_pos, spoon_local_grasp_rot, spoon_local_grasp_pos)

    global_franka_rot_1, global_franka_pos_1 = tf_combine(
        hand_rot_1, hand_pos_1, franka_local_grasp_rot_1, franka_local_grasp_pos_1)
    global_cup_rot, global_cup_pos = tf_combine(
        cup_rot, cup_pos, cup_local_grasp_rot, cup_local_grasp_pos)

    return global_franka_rot, global_franka_pos, global_spoon_rot, global_spoon_pos, global_franka_rot_1, global_franka_pos_1, global_cup_rot, global_cup_pos
