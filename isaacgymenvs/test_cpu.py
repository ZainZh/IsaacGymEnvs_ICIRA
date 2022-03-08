

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


# Parse arguments
args = gymutil.parse_arguments(description="Franka Tensor OSC Example",
                               custom_parameters=[
                                   {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
                                   {"name": "--pos_control", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Trace circular path in XZ plane"},
                                   {"name": "--orn_control", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Send random orientation commands"}])

# Initialize gym
gym = gymapi.acquire_gym()

# configure sim
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

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    raise Exception("Failed to create sim")

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 1, 0)
gym.add_ground(sim, plane_params)

# Load franka asset
asset_root = "../assets"
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
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)
franka_asset_1 = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)
# load table asset
cup_asset_options = gymapi.AssetOptions()
table_dims = gymapi.Vec3(2.4, 2.0, 3.0)
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
cup_asset = gym.load_asset(sim, asset_root, cup_asset_file, cup_asset_options)
cup_asset_options.fix_base_link = True
shelf_asset = gym.load_asset(sim, asset_root, shelf_asset_file, cup_asset_options)
# load cup and spoon

asset_options.fix_base_link = False
asset_options.disable_gravity = False

# cup_asset_options.fix_base_link = False

spoon_asset = gym.load_asset(sim, asset_root, spoon_asset_file, asset_options)

# create pose
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0, 0, 0)

pose = gymapi.Transform()

pose.p.x = table_pose.p.x - 0.3
pose.p.y = 1
pose.p.z = 0.29
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

pose_1 = gymapi.Transform()
pose_1.p.x = table_pose.p.x - 0.3
pose_1.p.y = 1
pose_1.p.z = -0.29

pose_1.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

cup_pose = gymapi.Transform()
cup_pose.p.x = table_pose.p.x + 0.3
cup_pose.p.y = 1.0
cup_pose.p.z = 0.29
cup_pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

spoon_pose = gymapi.Transform()
spoon_pose.p.x = table_pose.p.x + 0.25
spoon_pose.p.y = 1.107
spoon_pose.p.z = -0.29
spoon_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

shelf_pose = gymapi.Transform()
shelf_pose.p.x = table_pose.p.x + 0.3
shelf_pose.p.y = 1
shelf_pose.p.z = -0.29
shelf_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# Set up the env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, 0.0,-spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)



print("Creating %d environments" % num_envs)

envs = []
hand_idxs = []
init_pos_list = []
init_orn_list = []

for i in range(num_envs):
    # Create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    franka_actor = gym.create_actor(env, franka_asset, pose, "franka", i, 0)
    franka_actor_1 = gym.create_actor(env, franka_asset_1, pose_1, "franka1", i, 0)
    table_actor = gym.create_actor(env, table_asset, table_pose, "table", i, 0)
    cup_actor= gym.create_actor(env, cup_asset, cup_pose, "cup", i, 0)
    spoon_actor = gym.create_actor(env, spoon_asset, spoon_pose, "spoon", i, 0)
    shelf = gym.create_actor(env, shelf_asset, shelf_pose, "shelf", i, 0)
# Point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 3)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API to access and control the physics simulation
gym.prepare_sim(sim)





while not gym.query_viewer_has_closed(viewer):
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
