## CQL in IsaacGym

Complete [https://github.com/wzqwwq/IsaacGymEnvs](https://github.com/wzqwwq/IsaacGymEnvs) 

# INSTALLATION

1. The essential package [rl_games](https://github.com/Denys88/rl_games) use the newest version v1.3.2, have to download the package manually.

   Copy to path `/IsaacGymEnvs/issacgymenvs/rl_games`

2. Replace `isaacgymenvs` folder in `/IsaacGymEnvs/isaacgymenvs`

### Config

1. The config `DualFrankaCQL.yaml` is under `/IsaacGymEnvs/isaacgymenvs/cfg/train` 

2. Default argument `train: ${task}CQL` instead of `PPO` in `config.yaml` so that the program switch to read `DualFrankaCQL.yaml`

### Main 

Add `cql_agent.py` to do CQL.

# RUN

#### Train

```
python train.py num_envs=4096
```

#### Test

```
python train.py test=True headless=False checkpoint='runs/DualFrankaCQL/nn/model.pth'
```

# Run the DualFranka test program

1. Place `test_new.py` under the same folder with `train.py`, should be `/IsaacGymEnvs/isaacgymenvs`

2. Check the test config file in `test_config.ini`

3. Running in the environment

   ```
   python test.py
   ```

4. Save pose data under folder `./test_save`, hdf5 under `./test_save/hdf5`

### Comments

| Keyboard shortcut       | Description                                                  |
| :---------------------- | ------------------------------------------------------------ |
| `R`                     | Reset the environment. Need to disable pose override in the viewer first. |
| `T`                     | Print the obs/reward once in terminal.                       |
| `P`                     | Turn on/off the print info.                                  |
| `C`                     | Switch the view angle                                        |
| `S`                     | Save the pose status in file.                                |
| `0`                     | Enable/disable target tracking                               |
| `N`                     | Force jump to next stage in auto tracking                    |
| `D`                     | Enable/disable debug info                                    |
| `UP, DOWN, LEFT, RIGHT` | Drive franka position manually in `xz` plane                 |
| `[`, `]`                | Drive franka position manually in `y` axis                   |
| `LEFT_SHIFT`            | Switch driving between left/right Franka                     |

