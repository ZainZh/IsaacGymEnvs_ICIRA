import math
import numpy as np
import os
import torch
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
        self.num_agents = self.cfg["env"]["numAgents"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.ResetFromReplay = self.cfg["env"]["ResetFromReplay"]

        self.up_axis = "y"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.

        # prop dimensions
        self.prop_width = 0.06
        self.prop_height = 0.06
        self.prop_length = 0.06
        self.prop_spacing = 0.09
        self.gripped = torch.zeros((1, self.num_Envs))
        self.gripped_1 = torch.zeros((1, self.num_Envs))
        # num_obs = 42
        # num_acts = 18
        actors_per_env = 7
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
        self.cup_positions = self.root_tensor[..., 2, 0:3]
        self.cup_orientations = self.root_tensor[..., 2, 3:7]
        self.cup_linvels = self.root_tensor[..., 2, 7:10]
        self.cup_angvels = self.root_tensor[..., 2, 10:13]
        self.spoon_positions = self.root_tensor[..., 3, 0:3]
        self.spoon_orientations = self.root_tensor[..., 3, 3:7]
        self.spoon_linvels = self.root_tensor[..., 3, 7:10]
        self.spoon_angvels = self.root_tensor[..., 3, 10:13]
        self.shelf_positions = self.root_tensor[..., 4, 0:3]
        self.shelf_orientations = self.root_tensor[..., 4, 3:7]
        self.shelf_linvels = self.root_tensor[..., 4, 7:10]
        self.shelf_angvels = self.root_tensor[..., 4, 10:13]
        self.table_positions = self.root_tensor[..., 5, 0:3]
        self.table_orientations = self.root_tensor[..., 5, 3:7]
        self.table_linvels = self.root_tensor[..., 5, 7:10]
        self.table_angvels = self.root_tensor[..., 5, 10:13]
        # self.all_actor_indices = torch.arange(actors_per_env * self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs, actors_per_env)

        # create some wrapper tensors for different slices
        # set default pos (seven links and left and right hand)
        self.franka_default_dof_pos = to_torch([0.3863, 0.5062, -0.1184, -0.6105, 0.023, 1.6737, 0.9197, 0.04, 0.04],
                                               device=self.device)
        self.franka_default_dof_pos_1 = to_torch([-0.5349, 0, 0.1401, -1.7951, 0.0334, 3.2965, 0.6484, 0.04, 0.04],
                                                 device=self.device)
        self.franka_default_dof_pos_stage2 = to_torch([-0.3545, 0.6990, 0.2934, 0.7159, -0.1442, 0.3583, 0.5817, 0.04, 0.04],
                                               device=self.device)
        self.franka_default_dof_pos_1_stage2 = to_torch([-0.4901, 0.5585, -0.3762, -0.5309, -0.1273, -0.7353, -0.4016, 0.04, 0.04],
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

        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_state_1 = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:]

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
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.global_indices = torch.arange(self.num_envs * 7, dtype=torch.int32, device=self.device).view(self.num_envs,
                                                                                                          -1)
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
        spoon_asset_file = 'urdf/spoon/urdf/spoon.urdf'
        shelf_asset_file = 'urdf/shelf/urdf/shelf.urdf'

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

        # load table, cup asset
        table_dims = gymapi.Vec3(1, 0.4, 1.5)
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        # shelf_dims = gymapi.Vec3(0.15, 0.2, 0.15)
        # shelf_asset = self.gym.create_box(self.sim, shelf_dims.x, shelf_dims.y, shelf_dims.z, asset_options)
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

        # other_asset_options = gymapi.AssetOptions()
        # cup_asset = self.gym.load_asset(self.sim, asset_root, spoon_asset_file, other_asset_options)
        #
        # # load shelf and spoon
        # box_dims = gymapi.Vec3(0.1, 0.04, 0.1)
        # other_asset_options.fix_base_link = True
        # shelf_asset = self.gym.create_box(self.sim, box_dims.x, box_dims.y, box_dims.z, other_asset_options)
        # box_asset = self.gym.load_asset(self.sim, asset_root, shelf_asset_file, other_asset_options)
        # other_asset_options.fix_base_link = False
        # spoon_asset = self.gym.load_asset(self.sim, asset_root, cup_asset_file, other_asset_options)

        # box_opts = gymapi.AssetOptions()
        # box_opts.density = 400
        # prop_asset = self.gym.create_box(self.sim, self.prop_width, self.prop_height, self.prop_width, box_opts)

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
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        franka_dof_props_1 = self.gym.get_asset_dof_properties(franka_asset_1)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
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

        # create pose
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.5 * table_dims.y, 0.0)

        pose = gymapi.Transform()
        pose.p.x = -1
        if turn_spoon:
            pose.p.x = -1.2
        pose.p.y = 0.0
        pose.p.z = 0.5
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        pose_1 = gymapi.Transform()
        pose_1.p.x = -1
        pose_1.p.y = 0.0
        pose_1.p.z = -0.6
        pose_1.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        box_pose = gymapi.Transform()
        box_pose.p.x = table_pose.p.x - 0.3
        box_pose.p.y = table_pose.p.y + 0.5 * table_dims.y + 0.5 * box_dims.y
        box_pose.p.z = -0.29
        box_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        cup_pose = gymapi.Transform()
        cup_pose.p.x = table_pose.p.x - 0.3
        cup_pose.p.y = box_pose.p.y + 0.5 * box_dims.y
        cup_pose.p.z = -0.29
        cup_pose.r = gymapi.Quat(0.0, -0.287, 0.0, 0.95793058)

        spoon_pose = gymapi.Transform()
        spoon_pose.p.x = table_pose.p.x - 0.29
        spoon_pose.p.y = 0.5
        if spoon_as_box:
            spoon_pose.p.y = 0.5 + 0.5 * spoon_box_dims.y
        spoon_pose.p.z = 0.29
        if turn_spoon:
            spoon_pose.p.x = table_pose.p.x - 0.45
            spoon_pose.p.z = 0.39
        spoon_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        if turn_spoon:
            spoon_pose.r = gymapi.Quat(0.0, -0.707, 0.0, 0.707)

        shelf_pose = gymapi.Transform()
        shelf_pose.p.x = table_pose.p.x - 0.3
        shelf_pose.p.y = 0.4
        if turn_spoon:
            shelf_pose.p.x = table_pose.p.x - 0.45
        if shelf_as_box:
            shelf_pose.p.y = 0.4 + 0.5 * shelf_box_dims.y
        shelf_pose.p.z = 0.29
        shelf_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # box_pose = gymapi.Transform()
        # box_pose.p.x = table_pose.p.x - 0.3
        # box_pose.p.y = 0.4
        # box_pose.p.z = -0.29
        # box_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        #
        # cup_pose = gymapi.Transform()
        # cup_pose.p.x = table_pose.p.x - 0.3
        # cup_pose.p.y = box_pose.p.y + 0.5 * box_dims.y
        # cup_pose.p.z = -0.29
        # cup_pose.r = gymapi.Quat(0.0, -0.287, 0.0, 0.95793058)
        #
        # spoon_pose = gymapi.Transform()
        # spoon_pose.p.x = table_pose.p.x - 0.29
        # spoon_pose.p.y = 0.5
        # spoon_pose.p.z = 0.29
        # spoon_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        #
        # shelf_pose = gymapi.Transform()
        # shelf_pose.p.x = table_pose.p.x - 0.3
        # shelf_pose.p.y = table_pose.p.y + 0.5 * table_dims.y + 0.5 * box_dims.y
        # shelf_pose.p.z = 0.29
        # shelf_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
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

        self.frankas = []
        self.frankas_1 = []
        self.table = []
        self.spoon = []
        self.cup = []
        self.shelf = []
        # prop means spoon add cup
        # self.default_spoon_states = []
        # self.default_cup_states=[]
        # self.prop_start = []
        self.envs = []

        for i in range(self.num_envs):

            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            franka_actor = self.gym.create_actor(env_ptr, franka_asset, pose, "franka", i, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)
            franka_actor_1 = self.gym.create_actor(env_ptr, franka_asset_1, pose_1, "franka1", i, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor_1, franka_dof_props_1)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            cup_actor = self.gym.create_actor(env_ptr, cup_asset, cup_pose, "cup", i, 0)
            spoon_actor = self.gym.create_actor(env_ptr, spoon_asset, spoon_pose, "spoon", i, 0)
            shelf_actor = self.gym.create_actor(env_ptr, shelf_asset, shelf_pose, "shelf", i, 0)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
            box_actor = self.gym.create_actor(env_ptr, box_asset, box_pose, "box", i, 0)
            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.frankas_1.append(franka_actor_1)
            self.table.append(table_actor)
            self.cup.append(cup_actor)
            self.spoon.append(spoon_actor)
            self.shelf.append(shelf_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_link7")
        self.hand_handle_1 = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor_1, "panda_link7")
        self.table_handle = self.gym.find_actor_rigid_body_handle(env_ptr, table_actor, "table")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger")
        self.lfinger_handle_1 = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor_1, "panda_leftfinger")
        self.rfinger_handle_1 = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor_1, "panda_rightfinger")
        self.cup_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cup_actor, "base_link")
        self.spoon_handle = self.gym.find_actor_rigid_body_handle(env_ptr, spoon_actor, "base_link")
        self.shelf_handle = self.gym.find_actor_rigid_body_handle(env_ptr, shelf_actor, "base_link")
        if spoon_as_box:
            self.spoon_handle = self.gym.find_actor_rigid_body_handle(env_ptr, spoon_actor, "box")
        if shelf_as_box:
            self.shelf_handle = self.gym.find_actor_rigid_body_handle(env_ptr, shelf_actor, "box")

        self.init_data()
        # self.gym.get_sim_rigid_body_states(self.sim,gymapi.STATE_ALL)

    def init_data(self):  # define some init data for simulation
        # get franka data form
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_link7")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_leftfinger")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_rightfinger")

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
        hand_1 = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas_1[0], "panda_link7")
        lfinger_1 = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas_1[0], "panda_leftfinger")
        rfinger_1 = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas_1[0], "panda_rightfinger")

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
        cup_local_grasp_pose.p.y = 0.05  # half of the cup height
        cup_local_grasp_pose.p.z = 0
        cup_local_grasp_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.cup_local_grasp_pos = to_torch([cup_local_grasp_pose.p.x, cup_local_grasp_pose.p.y,
                                             cup_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        # print(self.cup_local_grasp_pos)
        self.cup_local_grasp_rot = to_torch([cup_local_grasp_pose.r.x, cup_local_grasp_pose.r.y,
                                             cup_local_grasp_pose.r.z, cup_local_grasp_pose.r.w],
                                            device=self.device).repeat((self.num_envs, 1))

        spoon_local_grasp_pose = gymapi.Transform()
        spoon_local_grasp_pose.p.x = 0.03
        spoon_local_grasp_pose.p.y = 0.005
        if spoon_as_box:
            spoon_local_grasp_pose.p.x = 0.03
            spoon_local_grasp_pose.p.y = 0.0
        spoon_local_grasp_pose.p.z = 0
        spoon_local_grasp_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

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
        if turn_spoon:
            self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))  # +z
            self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))  # +y

        # print('self.gripper_forward_axis: {}, self.gripper_up_axis: {}'.format(self.gripper_forward_axis,
        #                                                                         self.gripper_up_axis))
        self.cup_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))

        self.gripper_forward_axis_1 = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.spoon_inward_axis = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis_1 = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.spoon_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        if turn_spoon:
            self.spoon_inward_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs, 1))  # -z
            self.spoon_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))  # +y
        # print('self.spoon_inward_axis: {}, self.gripper_up_axis: {}'.format(self.spoon_inward_axis,
        #                                                                         self.spoon_up_axis))
        self.table_rot= to_torch([0, 0, 0, 1], device=self.device).repeat((self.num_envs, 1))

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
    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
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
        self.obs_buf = torch.cat((dof_pos_scaled, dof_pos_scaled_1,
                                  self.franka_dof_vel * self.dof_vel_scale, to_target,
                                  self.franka_dof_vel_1 * self.dof_vel_scale, to_target_1,
                                  spoon_pos, spoon_rot, cup_pos, cup_rot,
                                  self.franka_dof_pos, self.franka_dof_pos_1), dim=-1)

        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.reward_dict, self.gripped, self.gripped_1 = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions,
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
            self.max_episode_length)
        # self.reset_num1 = torch.cat((self.contact_forces[:, 0:5, :], self.contact_forces[:, 6:7, :]), 1)
        # self.reset_num2 = torch.cat((self.contact_forces[:, 10:15, :], self.contact_forces[:, 16:17, :]), 1)
        # self.reset_num = torch.cat((self.reset_num1, self.reset_num2), dim=-1)
        # self.reset_numm = torch.cat((self.contact_forces[:, 0:7, :], self.contact_forces[:, 10:17, :]), 1)

    def reset_idx_replay_buffer(self, env_ids):
        import h5py
        with h5py.File('./reset_buffer/replay_buff.hdf5', 'r') as hdf:
            ls = list(hdf.keys())
            data = hdf.get('observations')
            dataset1 = np.array(data)  # get the obversation buffer from replay buffer
            # rand_idx = random.randrange(0, dataset1.shape[0], 1)
            rand_idx = 350
            rand_franka_cup_pos = dataset1[rand_idx, -9:]
            rand_franka_spoon_pos = dataset1[rand_idx, -18:-9]
            rand_cup_pos = dataset1[rand_idx, -25:-18]
            rand_spoon_pos = dataset1[rand_idx, -32:-25]
        pos = to_torch(rand_franka_spoon_pos, device=self.device)
        # print("pos is ", pos)
        # reset franka with "pos"
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        # reset franka1
        pos_1 = to_torch(rand_franka_cup_pos, device=self.device)
        self.franka_dof_pos_1[env_ids, :] = pos_1
        self.franka_dof_vel_1[env_ids, :] = torch.zeros_like(self.franka_dof_vel_1[env_ids])
        self.franka_dof_targets[env_ids, self.num_franka_dofs: 2 * self.num_franka_dofs] = pos_1

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
        actor_indices = self.global_indices[env_ids, 2:4].flatten()

        actor_indices_32 = actor_indices.to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_tensor),
                                                     gymtorch.unwrap_tensor(actor_indices_32), len(actor_indices_32))

        multi_env_ids = self.global_indices[env_ids, :2].flatten()
        multi_env_ids_int32 = multi_env_ids.to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0

        self.reset_buf[env_ids] = 0

    def reset_idx(self, env_ids):
        # reset franka
        # self.root_states[env_ids] = self.saved_root_tensor[env_ids]

        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 0.1 * (
                    torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # print("pos is ", pos)
        # reset franka with "pos"
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos
        
        # reset franka1
        pos_1 = tensor_clamp(
            self.franka_default_dof_pos_1.unsqueeze(0) + 0.1 * (
                    torch.rand((len(env_ids), self.num_franka_dofs_1), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.franka_dof_pos_1[env_ids, :] = pos_1
        self.franka_dof_vel_1[env_ids, :] = torch.zeros_like(self.franka_dof_vel_1[env_ids])
        self.franka_dof_targets[env_ids, self.num_franka_dofs: 2 * self.num_franka_dofs] = pos_1

        # reset cup
        self.cup_positions[env_ids, 0] = -0.3
        self.cup_positions[env_ids, 1] = 0.443
        self.cup_positions[env_ids, 2] = -0.29
        self.cup_orientations[env_ids, 0:3] = 0.0
        self.cup_orientations[env_ids, 1] = -0.287
        self.cup_orientations[env_ids, 3] = 0.95793058
        self.cup_linvels[env_ids] = 0.0
        self.cup_angvels[env_ids] = 0.0

        # reset spoon
        self.spoon_positions[env_ids, 0] = -0.29
        self.spoon_positions[env_ids, 1] = 0.5
        if spoon_as_box:
            self.spoon_positions[env_ids, 1] = 0.5 + 0.015
        self.spoon_positions[env_ids, 2] = 0.29
        if turn_spoon:
            self.spoon_positions[env_ids, 0] = -0.53
            self.spoon_positions[env_ids, 2] = 0.39
        self.spoon_orientations[env_ids, 0] = 0.0
        self.spoon_orientations[env_ids, 1] = 0.0
        self.spoon_orientations[env_ids, 2] = 0.0
        self.spoon_orientations[env_ids, 3] = 1.0
        if turn_spoon:
            self.spoon_orientations[env_ids, 0] = 0.0
            self.spoon_orientations[env_ids, 1] = -0.707
            self.spoon_orientations[env_ids, 2] = 0.0
            self.spoon_orientations[env_ids, 3] = 0.707
        self.spoon_angvels[env_ids] = 0.0
        self.spoon_linvels[env_ids] = 0.0

        # # reset cup
        # self.cup_positions[env_ids, 0] = -0.29
        # self.cup_positions[env_ids, 1] = 0.5
        # self.cup_positions[env_ids, 2] = -0.29
        # self.cup_orientations[env_ids, 0:3] = 0.0
        # self.cup_orientations[env_ids, 1] = -0.0
        # self.cup_orientations[env_ids, 3] = 1.0
        # self.cup_linvels[env_ids] = 0.0
        # self.cup_angvels[env_ids] = 0.0
        #
        # # reset spoon
        # self.spoon_positions[env_ids, 0] = -0.3
        # self.spoon_positions[env_ids, 1] = 0.44
        # self.spoon_positions[env_ids, 2] = 0.29
        # self.spoon_orientations[env_ids, 0:3] = 0.0
        # self.spoon_orientations[env_ids, 1] = 0.0
        # self.spoon_orientations[env_ids, 3] = 1.0
        # self.spoon_angvels[env_ids] = 0.0
        # self.spoon_linvels[env_ids] = 0.0

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
        actor_indices = self.global_indices[env_ids, 2:4].flatten()

        actor_indices_32 = actor_indices.to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_tensor),
                                                     gymtorch.unwrap_tensor(actor_indices_32), len(actor_indices_32))

        multi_env_ids = self.global_indices[env_ids, :2].flatten()
        multi_env_ids_int32 = multi_env_ids.to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0

        self.reset_buf[env_ids] = 0

    def reset_idx_1(self, env_ids):
        # reset franka
        # self.root_states[env_ids] = self.saved_root_tensor[env_ids]
        pos = tensor_clamp(
            self.franka_default_dof_pos_2.unsqueeze(0) + 0.25 * (
                    torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # print("pos is ", pos)
        # reset franka with "pos"
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        # reset franka1
        pos_1 = tensor_clamp(
            self.franka_default_dof_pos_3.unsqueeze(0) + 0.25 * (
                    torch.rand((len(env_ids), self.num_franka_dofs_1), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.franka_dof_pos_1[env_ids, :] = pos_1
        self.franka_dof_vel_1[env_ids, :] = torch.zeros_like(self.franka_dof_vel_1[env_ids])
        self.franka_dof_targets[env_ids, self.num_franka_dofs:2 * self.num_franka_dofs] = pos_1

        # reset cup
        self.cup_positions[env_ids, 0] = -0.3
        self.cup_positions[env_ids, 1] = 0.443
        self.cup_positions[env_ids, 2] = -0.29
        self.cup_orientations[env_ids, 0:3] = 0.0
        self.cup_orientations[env_ids, 1] = -0.287
        self.cup_orientations[env_ids, 3] = 0.95793058
        self.cup_linvels[env_ids] = 0.0
        self.cup_angvels[env_ids] = 0.0

        # reset spoon
        self.spoon_positions[env_ids, 0] = -0.29
        self.spoon_positions[env_ids, 1] = 0.5
        self.spoon_positions[env_ids, 2] = 0.29
        self.spoon_orientations[env_ids, 0] = 0.0
        self.spoon_orientations[env_ids, 1] = 0.0
        self.spoon_orientations[env_ids, 2] = 0.0
        self.spoon_orientations[env_ids, 3] = 1.0
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
        actor_indices = self.global_indices[env_ids, 2:4].flatten()

        actor_indices_32 = actor_indices.to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_tensor),
                                                     gymtorch.unwrap_tensor(actor_indices_32), len(actor_indices_32))

        multi_env_ids = self.global_indices[env_ids, :2].flatten()
        multi_env_ids_int32 = multi_env_ids.to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0

        self.reset_buf[env_ids] = 0

    def reset_idx_2(self, env_ids):
        # reset franka
        # self.root_states[env_ids] = self.saved_root_tensor[env_ids]
        pos = tensor_clamp(
            self.franka_default_dof_pos_4.unsqueeze(0) + 0.25 * (
                    torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # print("pos is ", pos)
        # reset franka with "pos"
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        # reset franka1
        pos_1 = tensor_clamp(
            self.franka_default_dof_pos_5.unsqueeze(0) + 0.25 * (
                    torch.rand((len(env_ids), self.num_franka_dofs_1), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.franka_dof_pos_1[env_ids, :] = pos_1
        self.franka_dof_vel_1[env_ids, :] = torch.zeros_like(self.franka_dof_vel_1[env_ids])
        self.franka_dof_targets[env_ids, self.num_franka_dofs:2 * self.num_franka_dofs] = pos_1

        # reset cup
        # reset cup
        self.cup_positions[env_ids, 0] = -0.3
        self.cup_positions[env_ids, 1] = 0.443
        self.cup_positions[env_ids, 2] = -0.29
        self.cup_orientations[env_ids, 0:3] = 0.0
        self.cup_orientations[env_ids, 1] = -0.287
        self.cup_orientations[env_ids, 3] = 0.95793058
        self.cup_linvels[env_ids] = 0.0
        self.cup_angvels[env_ids] = 0.0

        # reset spoon
        self.spoon_positions[env_ids, 0] = -0.29
        self.spoon_positions[env_ids, 1] = 0.5
        self.spoon_positions[env_ids, 2] = 0.29
        self.spoon_orientations[env_ids, 0] = 0.0
        self.spoon_orientations[env_ids, 1] = 0.0
        self.spoon_orientations[env_ids, 2] = 0.0
        self.spoon_orientations[env_ids, 3] = 1.0
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
        actor_indices = self.global_indices[env_ids, 2:4].flatten()

        actor_indices_32 = actor_indices.to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_tensor),
                                                     gymtorch.unwrap_tensor(actor_indices_32), len(actor_indices_32))

        multi_env_ids = self.global_indices[env_ids, :2].flatten()
        multi_env_ids_int32 = multi_env_ids.to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0

        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)  # (num_envs, 18)

        # compute franka next target
        # relative control
        targets = self.franka_dof_targets[:,
                  :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions[:,
                                                                                    0:9] * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        targets_1 = self.franka_dof_targets[:,
                    self.num_franka_dofs: 2 * self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt \
                    * self.actions[:, 9:18] * self.action_scale
        self.franka_dof_targets[:, self.num_franka_dofs:2 * self.num_franka_dofs] = tensor_clamp(
            targets_1, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # grip_act_spoon=torch.where(self.gripped==1,torch.Tensor([[0.004, 0.004]] * self.num_envs).to(self.device), torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device))
        gripper_sep_spoon = self.franka_dof_targets[:, 7] + self.franka_dof_targets[:, 8]
        gripper_sep_cup = self.franka_dof_targets[:, -1] + self.franka_dof_targets[:, -2]
        # self.franka_dof_targets[:,7]=torch.where(gripper_sep_spoon<0.008,0.04, 0.005)
        # self.franka_dof_targets[:,8]=torch.where(gripper_sep_spoon<0.008,0.04, 0.005)
        self.franka_dof_targets[:, 7] = torch.where(self.gripped == 1, 0.0046, 0.04)
        self.franka_dof_targets[:, 8] = torch.where(self.gripped == 1, 0.0046, 0.04)
        self.franka_dof_targets[:, -1] = torch.where(self.gripped_1 == 1, 0.0244, 0.04)
        self.franka_dof_targets[:, -2] = torch.where(self.gripped_1 == 1, 0.0244, 0.04)
        # print(self.franka_dof_targets[:,7],self.franka_dof_targets[:,8])
        # give to gym
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))

    def post_physics_step(self):  # what do frankas do after interacting with the envs
        self.progress_buf += 1
        # print("progress buffer is ",self.progress_buf)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # print("env_ids", env_ids)

        # # choose the simulation way(multi ,single or resetfromreplay) and comment out the other way
        # env_ids_new = torch.zeros_like(env_ids)
        # env_ids_new_1 = torch.zeros_like(env_ids)
        # env_ids_new_2 = torch.zeros_like(env_ids)
        # j = 0
        # k = 0
        # m = 0
        # # multi simulation(different private pose)
        # if len(env_ids) > 0:
        #     for i in range(len(env_ids)):
        #         if env_ids[i] < self.num_Envs / 3:
        #             env_ids_new[j] = env_ids[i]
        #             j = j + 1
        #         if env_ids[i] >= self.num_Envs / 3:
        #             if env_ids[i] <2*self.num_Envs / 3:
        #                 env_ids_new_1[k] = env_ids[i]
        #                 k = k + 1
        #         if env_ids[i] >= 2*self.num_Envs / 3:
        #             if env_ids[i] < self.num_Envs :
        #                 env_ids_new_2[m] = env_ids[i]
        #                 m = m + 1
        #     if len(env_ids_new[:j]) > 0:
        #         self.reset_idx(env_ids_new[:j])
        #     if len(env_ids_new_1[:k]) > 0:
        #         self.reset_idx_1(env_ids_new_1[:k])
        #     if len(env_ids_new_2[:m]) > 0:
        #         self.reset_idx_2(env_ids_new_2[:m])
        #     print("env_ids_new", env_ids_new[:j])
        #     print("env_ids_neww", env_ids_new_1[:k])
        #     print("env_ids_newwW", env_ids_new_2[:m])

        # single simulation and reset from replay buffer

        if len(env_ids) > 0:
            if self.ResetFromReplay == True:
                self.reset_idx_replay_buffer(env_ids)
            else:
                self.reset_idx(env_ids)

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

                px = (self.spoon_grasp_pos[i] + quat_apply(self.spoon_grasp_rot[i],
                                                           to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.spoon_grasp_pos[i] + quat_apply(self.spoon_grasp_rot[i],
                                                           to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.spoon_grasp_pos[i] + quat_apply(self.spoon_grasp_rot[i],
                                                           to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

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
        reset_buf, progress_buf, actions, franka_grasp_pos, cup_grasp_pos, franka_grasp_rot, franka_grasp_pos_1,
        spoon_grasp_pos, franka_grasp_rot_1, cup_grasp_rot, spoon_grasp_rot,table_rot,
        cup_positions, spoon_positions, cup_orientations, spoon_orientations,
        spoon_linvels, cup_linvels,
        cup_inward_axis, cup_up_axis, franka_lfinger_pos, franka_rfinger_pos,
        spoon_inward_axis, spoon_up_axis, franka_lfinger_pos_1, franka_rfinger_pos_1,
        gripper_forward_axis, gripper_up_axis,
        gripper_forward_axis_1, gripper_up_axis_1, contact_forces,
        num_envs: int, dist_reward_scale: float, rot_reward_scale: float, around_handle_reward_scale: float,
        lift_reward_scale: float, finger_dist_reward_scale: float, action_penalty_scale: float, distX_offset: float,
        max_episode_length: float
) -> Tuple[Tensor, Tensor, Dict[str, Union[Dict[str, Tuple[Tensor, Union[Tensor, float]]], Dict[str, Tuple[Tensor, float]], Dict[str, Tensor]]], Tensor, Tensor]:
    """
    Tuple[Tensor, Tensor, Dict[str, Union[Dict[str, Tuple[Tensor, float]],
                                           Dict[str, Tensor], Dict[str, Union[Tensor, Tuple[Tensor, float]]]]]]:
    """
    tensor_device = franka_grasp_pos.device
    # turn_spoon = True

    # print('spoon_grasp_pos: {}'.format(spoon_grasp_pos[0]))
    # print('cup_grasp_pos: {}'.format(cup_grasp_pos[0]))
    # print('spoon_positions: {}'.format(spoon_positions[0]))
    # print('cup_positions: {}'.format(cup_positions[0]))
    # print('franka_lfinger_pos: {}'.format(franka_lfinger_pos[0]))
    # print('franka_rfinger_pos: {}'.format(franka_rfinger_pos[0]))
    # print('franka_lfinger_pos_1: {}'.format(franka_lfinger_pos_1[0]))
    # print('franka_rfinger_pos_1: {}'.format(franka_rfinger_pos_1[0]))

    # <editor-fold desc="1. distance reward - grasp and object">
    d = torch.norm(franka_grasp_pos - spoon_grasp_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.06, dist_reward * 2, dist_reward)
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)  # TODO: test

    d_1 = torch.norm(franka_grasp_pos_1 - cup_grasp_pos, p=2, dim=-1)
    dist_reward_1 = 1.0 / (1.0 + d_1 ** 2)
    dist_reward_1 *= dist_reward_1
    dist_reward_1 = torch.where(d_1 <= 0.06, dist_reward_1 * 2, dist_reward_1)
    # </editor-fold>

    # <editor-fold desc="2. rotation reward">
    # define axis to make sure the alignment
    axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)  # franka
    axis2 = tf_vector(spoon_grasp_rot, spoon_up_axis)  # [0,1,0]
    axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
    axis4 = tf_vector(spoon_grasp_rot, spoon_inward_axis)  # [1,0,0]

    axis1_1 = tf_vector(franka_grasp_rot_1, gripper_forward_axis_1)  # franka1
    axis2_1 = tf_vector(cup_grasp_rot, cup_inward_axis)
    axis3_1 = tf_vector(franka_grasp_rot_1, gripper_up_axis_1)
    axis4_1 = tf_vector(cup_grasp_rot, cup_up_axis)

    axistable=tf_vector(table_rot,cup_up_axis) #add ground axis

    # compute the alignment(reward)
    # alignment of forward axis for gripper
    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # franka-z-minus with spoon-y
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # franka-x with spoon-x

    '''Box and turn'''
    # dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(
    #     -1)  # franka-z with spoon-z-minus
    # dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(
    #     -1)  # franka-y with spoon-y

    dot1_1 = torch.bmm(axis1_1.view(num_envs, 1, 3), axis2_1.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # franka-z with cup-x
    dot2_1 = torch.bmm(axis3_1.view(num_envs, 1, 3), axis4_1.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # franka-x with cup-y

    # reward for matching the orientation of the hand to the cup(fingers wrapped)
    rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)
    rot_reward_1 = 0.5 * (torch.sign(dot1_1) * dot1_1 ** 2 + torch.sign(dot2_1) * dot2_1 ** 2)
    # </editor-fold>

    # <editor-fold desc="3. around reward">
    # bonus if right/L finger is at the object R/L side
    # around_handle_reward = torch.zeros_like(rot_reward)
    # around_handle_reward = torch.where(quat_rotate_inverse(spoon_grasp_rot, franka_lfinger_pos)[:, 2] \
    #                                    > quat_rotate_inverse(spoon_grasp_rot, spoon_grasp_pos)[:, 2],
    #                                    torch.where(quat_rotate_inverse(spoon_grasp_rot, franka_rfinger_pos)[:, 2] \
    #                                                < quat_rotate_inverse(spoon_grasp_rot, spoon_grasp_pos)[:, 2],
    #                                                around_handle_reward + 0.5, around_handle_reward),
    #                                    around_handle_reward)
    # around_handle_reward_1 = torch.zeros_like(rot_reward_1)
    # around_handle_reward_1 = torch.where(quat_rotate_inverse(cup_grasp_rot, franka_lfinger_pos_1)[:, 2] \
    #                                      > quat_rotate_inverse(cup_grasp_rot, cup_grasp_pos)[:, 2],
    #                                      torch.where(quat_rotate_inverse(cup_grasp_rot, franka_rfinger_pos_1)[:, 2] \
    #                                                  < quat_rotate_inverse(cup_grasp_rot, cup_grasp_pos)[:, 2],
    #                                                  around_handle_reward_1 + 0.5, around_handle_reward_1),
    #                                      around_handle_reward_1)

    '''Box and turn'''
    # around_handle_reward = torch.zeros_like(rot_reward)
    # around_handle_reward = torch.where(franka_lfinger_pos[:, 1] > spoon_grasp_pos[:, 1],
    #                                    torch.where(franka_rfinger_pos[:, 1] < spoon_grasp_pos[:, 1],
    #                                                around_handle_reward + 0.5, around_handle_reward),
    #                                    around_handle_reward)

    '''Continous'''
    # around_handle_reward = torch.zeros_like(rot_reward)
    # around_handle_reward = torch.where(franka_lfinger_pos[:, 2] > spoon_grasp_pos[:, 2],
    #                                    torch.where(franka_rfinger_pos[:, 2] < spoon_grasp_pos[:, 2],
    #                                                around_handle_reward + 0.5, around_handle_reward),
    #                                    around_handle_reward)
    #
    # around_handle_reward_1 = torch.zeros_like(rot_reward_1)
    # around_handle_reward_1 = torch.where(franka_lfinger_pos_1[:, 2] > cup_grasp_pos[:, 2],
    #                                      torch.where(franka_rfinger_pos_1[:, 2] < cup_grasp_pos[:, 2],
    #                                                  around_handle_reward_1 + 0.5, around_handle_reward_1),
    #                                      around_handle_reward_1)

    '''Discrete'''
    spoon_size = [0.1, 0.01, 0.01]
    around_handle_reward = torch.zeros_like(rot_reward)
    around_handle_reward = torch.where(quat_rotate_inverse(spoon_grasp_rot, franka_lfinger_pos)[:, 2] \
                                       > quat_rotate_inverse(spoon_grasp_rot, spoon_grasp_pos)[:, 2] - (
                                               0.5 * spoon_size[2] + 0.002),
                                       torch.where(quat_rotate_inverse(spoon_grasp_rot, franka_rfinger_pos)[:, 2] \
                                                   < quat_rotate_inverse(spoon_grasp_rot, spoon_grasp_pos)[:, 2] + (
                                                           0.5 * spoon_size[2] + 0.002),
                                                   around_handle_reward + 0.5, around_handle_reward),
                                       around_handle_reward)
    around_handle_reward = torch.where(quat_rotate_inverse(spoon_grasp_rot, franka_lfinger_pos)[:, 0] \
                                       > quat_rotate_inverse(spoon_grasp_rot, spoon_grasp_pos)[:, 0] - (
                                               0.5 * spoon_size[0] + 0.002),
                                       torch.where(quat_rotate_inverse(spoon_grasp_rot, franka_rfinger_pos)[:, 0] \
                                                   < quat_rotate_inverse(spoon_grasp_rot, spoon_grasp_pos)[:, 0] + (
                                                           0.5 * spoon_size[0] + 0.002),
                                                   around_handle_reward + 0.5, around_handle_reward),
                                       around_handle_reward)
    around_handle_reward = torch.where(quat_rotate_inverse(spoon_grasp_rot, franka_lfinger_pos)[:, 1] \
                                       > quat_rotate_inverse(spoon_grasp_rot, spoon_grasp_pos)[:, 1] - (
                                               0.5 * spoon_size[1] + 0.002),
                                       torch.where(quat_rotate_inverse(spoon_grasp_rot, franka_rfinger_pos)[:, 1] \
                                                   < quat_rotate_inverse(spoon_grasp_rot, spoon_grasp_pos)[:, 1] + (
                                                           0.5 * spoon_size[1] + 0.002),
                                                   around_handle_reward + 0.5, around_handle_reward),
                                       around_handle_reward)
    gripped = (around_handle_reward == 1.5)
    cup_size = [0.05, 0.1, 0.05]
    around_handle_reward_1 = torch.zeros_like(rot_reward_1)
    around_handle_reward_1 = torch.where(quat_rotate_inverse(cup_grasp_rot, franka_lfinger_pos_1)[:, 2] \
                                         > quat_rotate_inverse(cup_grasp_rot, cup_grasp_pos)[:, 2] - (
                                                 0.5 * cup_size[2] + 0.002),
                                         torch.where(quat_rotate_inverse(cup_grasp_rot, franka_rfinger_pos_1)[:, 2] \
                                                     < quat_rotate_inverse(cup_grasp_rot, cup_grasp_pos)[:, 2] + (
                                                             0.5 * cup_size[2] + 0.002),
                                                     around_handle_reward_1 + 0.5, around_handle_reward_1),
                                         around_handle_reward_1)
    around_handle_reward_1 = torch.where(quat_rotate_inverse(cup_grasp_rot, franka_lfinger_pos_1)[:, 0] \
                                         > quat_rotate_inverse(cup_grasp_rot, cup_grasp_pos)[:, 0] - (
                                                 0.5 * cup_size[0] + 0.002),
                                         torch.where(quat_rotate_inverse(cup_grasp_rot, franka_rfinger_pos_1)[:, 0] \
                                                     < quat_rotate_inverse(cup_grasp_rot, cup_grasp_pos)[:, 0] + (
                                                             0.5 * cup_size[0] + 0.002),
                                                     around_handle_reward_1 + 0.5, around_handle_reward_1),
                                         around_handle_reward_1)
    around_handle_reward_1 = torch.where(quat_rotate_inverse(cup_grasp_rot, franka_lfinger_pos_1)[:, 1] \
                                         > quat_rotate_inverse(cup_grasp_rot, cup_grasp_pos)[:, 1] - (
                                                 0.5 * cup_size[1] - 0.01),
                                         torch.where(quat_rotate_inverse(cup_grasp_rot, franka_rfinger_pos_1)[:, 1] \
                                                     < quat_rotate_inverse(cup_grasp_rot, cup_grasp_pos)[:, 1] + (
                                                             0.5 * cup_size[1] - 0.01),
                                                     around_handle_reward_1 + 0.5, around_handle_reward_1),
                                         around_handle_reward_1)
    gripped_1 = (around_handle_reward_1 == 1.5)
    # </editor-fold>

    # <editor-fold desc="4. reward for distance of each finger from the objects">
    # XXX: inital leftfranka z-pos is near cup-z already, distance reward seems like around reward
    # reward for distance of each finger from the spoon, finger distance=0.08
    # finger_dist_reward = torch.zeros_like(rot_reward)
    # lfinger_dist = quat_rotate_inverse(spoon_grasp_rot, franka_lfinger_pos - spoon_grasp_pos)[:, 2]
    # rfinger_dist = quat_rotate_inverse(spoon_grasp_rot, franka_rfinger_pos - spoon_grasp_pos)[:, 2]
    # finger_dist_reward = torch.where(lfinger_dist > 0,
    #                                  torch.where(rfinger_dist < 0,
    #                                              50 * (0.08 - (torch.abs(lfinger_dist) + torch.abs(rfinger_dist))),
    #                                              finger_dist_reward),
    #                                  finger_dist_reward)
    #
    # # reward for distance of each finger from the cup
    # finger_dist_reward_1 = torch.zeros_like(rot_reward_1)
    # lfinger_dist_1 = quat_rotate_inverse(cup_grasp_rot, franka_lfinger_pos_1 - cup_grasp_pos)[:, 2]
    # rfinger_dist_1 = quat_rotate_inverse(cup_grasp_rot, franka_rfinger_pos_1 - cup_grasp_pos)[:, 2]
    # finger_dist_reward_1 = torch.where(lfinger_dist_1 > 0,
    #                                    torch.where(rfinger_dist_1 < 0,
    #                                                50 * (0.08 - 0.2 * (
    #                                                        torch.abs(lfinger_dist_1) + torch.abs(rfinger_dist_1))),
    #                                                finger_dist_reward_1),
    #                                    finger_dist_reward_1)

    '''Box and turn'''
    # finger_dist_reward = torch.zeros_like(rot_reward)
    # lfinger_dist = torch.abs(franka_lfinger_pos[:, 1] - spoon_grasp_pos[:, 1])
    # rfinger_dist = torch.abs(franka_rfinger_pos[:, 1] - spoon_grasp_pos[:, 1])
    # finger_dist_reward = torch.where(franka_lfinger_pos[:, 1] > spoon_grasp_pos[:, 1],
    #                                  torch.where(franka_rfinger_pos[:, 1] < spoon_grasp_pos[:, 1],
    #                                              (0.04 - lfinger_dist) + (0.04 - rfinger_dist), finger_dist_reward),
    #                                  finger_dist_reward)  # 2 together with the following lines
    #
    # '''Important'''
    # finger_dist_reward = torch.where(franka_lfinger_pos[:, 1] > spoon_grasp_pos[:, 1],
    #                                  torch.where(franka_rfinger_pos[:, 1] < spoon_grasp_pos[:, 1], torch.where(
    #                                      d <= 0.02, ((0.04 - lfinger_dist) + (0.04 - rfinger_dist)) * 100,
    #                                      finger_dist_reward), finger_dist_reward),
    #                                  finger_dist_reward)  # 3

    '''Continous'''
    # finger_dist_reward = torch.zeros_like(rot_reward)
    # lfinger_dist = torch.abs(franka_lfinger_pos[:, 2] - (spoon_grasp_pos[:, 2]))
    # rfinger_dist = torch.abs(franka_rfinger_pos[:, 2] - (spoon_grasp_pos[:, 2]))
    # finger_dist_reward = torch.where(franka_lfinger_pos[:, 2] > spoon_grasp_pos[:, 2],
    #                                  torch.where(franka_rfinger_pos[:, 2] < spoon_grasp_pos[:, 2],
    #                                              (0.04 - lfinger_dist) + (0.04 - rfinger_dist), finger_dist_reward),
    #                                  finger_dist_reward)
    # finger_dist_reward = torch.where(franka_lfinger_pos[:, 2] > spoon_grasp_pos[:, 2],
    #                                  torch.where(franka_rfinger_pos[:, 2] < spoon_grasp_pos[:, 2], torch.where(
    #                                      d <= 0.02, ((0.04 - lfinger_dist) + (0.04 - rfinger_dist)) * 100,
    #                                      finger_dist_reward), finger_dist_reward),
    #                                  finger_dist_reward)
    #
    # finger_dist_reward_1 = torch.zeros_like(rot_reward_1)
    # lfinger_dist_1 = torch.abs(franka_lfinger_pos_1[:, 2] - (cup_grasp_pos[:, 2]))
    # rfinger_dist_1 = torch.abs(franka_rfinger_pos_1[:, 2] - (cup_grasp_pos[:, 2]))
    # finger_dist_reward_1 = torch.where(franka_lfinger_pos_1[:, 2] > cup_grasp_pos[:, 2],
    #                                    torch.where(franka_rfinger_pos_1[:, 2] < cup_grasp_pos[:, 2],
    #                                                (0.04 - lfinger_dist_1) + (0.04 - rfinger_dist_1),
    #                                                finger_dist_reward_1),
    #                                    finger_dist_reward_1)
    # finger_dist_reward_1 = torch.where(franka_lfinger_pos_1[:, 2] > cup_grasp_pos[:, 2],
    #                                    torch.where(franka_rfinger_pos_1[:, 2] < cup_grasp_pos[:, 2], torch.where(
    #                                        d_1 <= 0.02, ((0.04 - lfinger_dist_1) + (0.04 - rfinger_dist_1)) * 100,
    #                                        finger_dist_reward_1), finger_dist_reward_1),
    #                                    finger_dist_reward_1)  # 3

    '''Discrete'''
    finger_dist_reward = torch.zeros_like(rot_reward)
    lfinger_dist = quat_rotate_inverse(spoon_grasp_rot, franka_lfinger_pos - spoon_grasp_pos)[:, 2]
    rfinger_dist = quat_rotate_inverse(spoon_grasp_rot, franka_rfinger_pos - spoon_grasp_pos)[:, 2]
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
    lfinger_dist_1 = quat_rotate_inverse(cup_grasp_rot, franka_lfinger_pos_1 - cup_grasp_pos)[:, 2]
    rfinger_dist_1 = quat_rotate_inverse(cup_grasp_rot, franka_rfinger_pos_1 - cup_grasp_pos)[:, 2]
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
                                                   finger_dist_reward_1),finger_dist_reward_1),
                                       finger_dist_reward_1)
    # </editor-fold>

    # <editor-fold desc="5. fall penalty(table or ground)">
    # # cup(fall and reverse)
    # cup_fall_penalty = torch.where(cup_positions[:, 1] < 0.439, 1.0, 0.0)
    dot_cup_reverse = torch.bmm(axis4_1.view(num_envs, 1, 3), cup_up_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # cup rotation y align with ground y(=cup up axis)
    # cup_reverse_penalty = torch.where(torch.acos(dot_cup_reverse) * 180 / torch.pi > 45 , 1.0, 0.0)    
    # spoon
    spoon_fall_penalty = torch.where(spoon_positions[:, 1] < 0.48, 1.0, 0.0)
    # </editor-fold>

    # <editor-fold desc="6. action penalty">
    # regularization on the actions (summed for each environment) (more actions more penalty)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    # </editor-fold>

    # <editor-fold desc="7. lift reward">
    # the higher the y coordinates of objects are, the larger the rewards will be set
    init_spoon_pos = torch.tensor([-0.2785, 0.499, 0.29])  # TODO: need to be changed
    init_cup_pos = torch.tensor([-0.3, 0.4425, -0.29])
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
    franka_grasp_pos_trans = franka_grasp_pos.t()
    franka_grasp_pos_1_trans = franka_grasp_pos_1.t()
    idx = 1
    franka_grasp_pos_stage2 = franka_grasp_pos_trans[torch.arange(franka_grasp_pos_trans.size(0)) != idx]
    franka_grasp_pos_1_stage2 = franka_grasp_pos_1_trans[torch.arange(franka_grasp_pos_1_trans.size(0)) != idx]
    d_spoon_cup = torch.norm(franka_grasp_pos_stage2.t() - franka_grasp_pos_1_stage2.t(), p=2, dim=-1)

    # dist_reward = 2.0 / (1.0 + d ** 2)
    dist_reward_stage2 = 1.0 / (1.0 + d_spoon_cup ** 2)
    dist_reward_stage2 *= dist_reward_stage2

    dot_stage2 = torch.bmm(axis4_1.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # spoon z with cup-y
    dot_stage2_table=torch.bmm(axistable.view(num_envs, 1, 3), axis4_1.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # spoon z with cup-y
    rot_reward_stage2 = 0.5*(torch.sign(dot_stage2) * dot_stage2 ** 2+torch.sign(dot_stage2_table) * dot_stage2_table ** 2)
    
    d_spoon_cup_y = torch.norm(franka_grasp_pos[:, 1] - franka_grasp_pos_1[:, 1], p=2, dim=-1)
    dist_reward_stage2_y=torch.zeros_like(dist_reward)
    # dist_reward_stage2_y = torch.where(franka_grasp_pos[:, 1] > franka_grasp_pos_1[:, 1],
    #                                    torch.where(d_spoon_cup_y > 0.4, 1.0 / (1.0 + d_spoon_cup_y ** 2),
    #                                                dist_reward_stage2_y),
    #                                    dist_reward_stage2_y)
    # dist_reward_stage2_y *= dist_reward_stage2_y

    # ....................stage 3 reward....................................................................
    preset_h = 0.09
    cup_r = 0.025
    spoon_tip_pos = quat_rotate_inverse(spoon_orientations,spoon_positions) - 0.5 * torch.tensor([0.15, 0.0, 0.0], device=tensor_device)
    spoon_tip_pos = quat_rotate(spoon_orientations,spoon_tip_pos)
    v1_s3 = quat_rotate_inverse(cup_orientations, spoon_tip_pos-cup_positions)   # relative spoon pos in cup
    prestage_s3 = [ torch.gt(torch.tensor([cup_r, cup_r], device=tensor_device),v1_s3[:, [0,2]]).all(dim=-1) ,
                    torch.lt(torch.tensor([-cup_r, -cup_r], device=tensor_device),v1_s3[:, [0,2]]).all(dim=-1) ,   # x,z in cup
                    torch.logical_and(v1_s3[:, 1] - preset_h < 0,v1_s3[:, 1] > 0) ]   # spoon tip in cup
    stage_s3 = torch.logical_and(v1_s3[:, 1] - 0.11 < 0,v1_s3[:, 1] > 0)
    flag_range_s3 = torch.logical_and(prestage_s3[0], prestage_s3[1])
    flag_full_s3 = torch.logical_and(flag_range_s3, prestage_s3[2])
    h_s3 = torch.abs(v1_s3[:, 1] - preset_h)
    h_reward_s3 = 2.0 / (1.0 + h_s3**2) * flag_range_s3
    d_s3 = torch.norm(v1_s3[:, [0,2]] - torch.tensor([cup_r, cup_r], device=tensor_device), dim=-1)
    d_reward_s3 = 2.0 / (1.0 + d_s3**2) * flag_range_s3
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
    stage1 = 1 # stage1 flag
    stage2 = 0  # stage2 flag
    stage3 = 0  # stage3 flag
    rewards = stage1*(dist_reward_scale * (dist_reward * sf + dist_reward_1 * cf) \
              + rot_reward_scale * (rot_reward * sf + rot_reward_1 * cf) \
              + around_handle_reward_scale * (around_handle_reward * sf + around_handle_reward_1 * cf) \
              + finger_dist_reward_scale * (finger_dist_reward * sf + finger_dist_reward_1 * cf) \
              + lift_reward_scale * (lift_reward * sf + lift_reward_1 * cf) \
              - action_penalty_scale * action_penalty \
              - spoon_fall_penalty)
              
    rewards = rewards + stage2 * (lift_reward_scale * 0.1 * (lift_reward * sf + lift_reward_1 * cf) \
                    + dist_reward_stage2 * dist_reward_scale * 20 \
                    + rot_reward_stage2 * rot_reward_scale * 20 \
                    + dist_reward_stage2_y * dist_reward_scale * 20)

    #TODO: add stage 3 reward
    fulfill_s2 = torch.logical_and(spoon_positions[:,1]-0.4 > 0.15,    # spoon_y - table_height > x  (shelf height ignored)
                    cup_positions[:,1]-0.4 > 0.15,    
                    )
    rewards = rewards + stage3 * fulfill_s2 *( h_reward_s3 * 20 \
                    + d_reward_s3 * 20 \
                    + v_reward_s3 * 20)

    # test args
    rewards_step = rewards.clone().detach()

    # <editor-fold desc="II. reward bonus">
    # <editor-fold desc="1. bonus for take up the cup properly(franka1)">
    # rewards = torch.where(spoon_positions[:, 1] > 0.51, rewards + 0.5, rewards)
    # rewards = torch.where(spoon_positions[:, 1] > 0.7, rewards + around_handle_reward, rewards)
    # rewards = torch.where(spoon_positions[:, 1] > 1.1, rewards + (2.0 * around_handle_reward), rewards)
    # # test args
    # take_spoon_bonus = rewards - rewards_step
    # rewards_step += take_spoon_bonus

    # rewards = torch.where(cup_positions[:, 1] > 0.45, rewards + 0.5, rewards)
    # rewards = torch.where(cup_positions[:, 1] > 0.64, rewards + around_handle_reward_1, rewards)
    # rewards = torch.where(cup_positions[:, 1] > 0.78, rewards + (2.0 * around_handle_reward_1), rewards)
    # # test args
    # take_cup_bonus = rewards - rewards_step
    # rewards_step += take_cup_bonus
    # </editor-fold>

    # <editor-fold desc="2. collision penalty (contact)">
    # ignore franka&franka1: link6, gripper force
    # reset_num1 = torch.cat((contact_forces[:, 0:5, :], contact_forces[:, 6:7, :]), dim=1)
    # reset_num2 = torch.cat((contact_forces[:, 10:15, :], contact_forces[:, 16:17, :]), dim=1)
    # # reset numm is the sum of the force in all of links which are unable to get force
    # reset_num = torch.cat((reset_num1, reset_num2), dim=-1)
    # reset_numm = torch.norm(torch.norm(reset_num[:, :, 0:6], dim=1), dim=1)  # value is contact force
    # # reward compute
    # rewards = torch.where(reset_numm > 600, rewards - 1, rewards)
    # # test args
    # collision_penalty_bonus = rewards - rewards_step
    # rewards_step += collision_penalty_bonus
    # </editor-fold>

    # <editor-fold desc="3.lossen gripper penalty(penalty when cup or spoon is taken up but gripper lossen.)">
    # rewards = torch.where(cup_positions[:, 1] > 0.445,
    #                       torch.where(torch.abs(lfinger_dist_1) + torch.abs(rfinger_dist_1) > 0.055,
    #                                   # give +0.005 tolerance
    #                                   rewards - 0.1,
    #                                   rewards), rewards)
    # cup_penalty_lossen = rewards - rewards_step
    # rewards_step += cup_penalty_lossen
    #
    # rewards = torch.where(spoon_positions[:, 1] > 0.5,
    #                       torch.where(torch.abs(lfinger_dist) + torch.abs(rfinger_dist) > 0.015,
    #                                   rewards - 0.1,
    #                                   rewards), rewards)
    # spoon_penalty_lossen = rewards - rewards_step
    # rewards_step += spoon_penalty_lossen
    # </editor-fold>
    # </editor-fold>

    # <editor-fold desc="prevent bad style in catching cup (opposite orientation)">
    # rewards = torch.where(franka_lfinger_pos[:, 0] < cup_grasp_pos[:, 0] - distX_offset,
    #                       torch.ones_like(rewards) * -1, rewards)
    # rewards = torch.where(franka_rfinger_pos[:, 0] < cup_grasp_pos[:, 0] - distX_offset,
    #                       torch.ones_like(rewards) * -1, rewards)
    # rewards = torch.where(franka_lfinger_pos_1[:, 0] < spoon_grasp_pos[:, 0] - distX_offset,
    #                       torch.ones_like(rewards) * -1, rewards)
    # rewards = torch.where(franka_rfinger_pos_1[:, 0] < spoon_grasp_pos[:, 0] - distX_offset,
    #                       torch.ones_like(rewards) * -1, rewards)

    # print("reward is ", rewards)
    # </editor-fold>

    # <editor-fold desc="Reset">
    # if reset buf equal to 1, reset this environment
    # reset if cup and spoon is taken up (max) or max length reached
    '''taken up too high'''
    reset_buf = torch.where(spoon_positions[:, 1] > 1.1, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(cup_positions[:, 1] > 0.78, torch.ones_like(reset_buf), reset_buf)  #
    '''fall'''
    reset_buf = torch.where(cup_positions[:, 1] < 0.3, torch.ones_like(reset_buf),
                            reset_buf)  # cup fall to table or ground
    reset_buf = torch.where(torch.acos(dot_cup_reverse) * 180 / torch.pi > 90, torch.ones_like(reset_buf),
                            reset_buf)  # cup fall direction
    reset_buf = torch.where(spoon_positions[:, 1] < 0.418, torch.ones_like(reset_buf),
                            reset_buf)  # spoon fall to table or ground

    # # cup fall to table
    # reset_buf = torch.where(reset_numm > 400, torch.ones_like(reset_buf), reset_buf)
    # reset when max_episode_length
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
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
        # 'take_cup_bonus(franka0)': take_cup_bonus,
        # # 'collision_penalty_bonus': collision_penalty_bonus,
        # # 'cup_penalty_lossen': cup_penalty_lossen,
        # # 'spoon_penalty_lossen': spoon_penalty_lossen,
        # 'take_spoon_bonus': take_spoon_bonus,
        # 'take_cup_bonus': take_cup_bonus,
        # 'collision_penalty_bonus': collision_penalty_bonus,
        # 'cup_penalty_lossen': cup_penalty_lossen,
        # 'spoon_penalty_lossen': spoon_penalty_lossen,
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

    return rewards, reset_buf, rewards_dict, gripped, gripped_1


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