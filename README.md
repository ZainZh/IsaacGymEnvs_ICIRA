# IsaacGym Bimanual Franka task

This repo is based on the Nvidia's repo [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs), editted by me for the ICIRA conference and RAS conference experiment.

The paper for ICIRA is accepted, link will be put as soon as possible.
- [IsaacGym Bimanual Franka task](#isaacgym-bimanual-franka-task)
	- [Handbook about our work](#handbook-about-our-work)
		- [Installation](#installation)
		- [Test environment during the training](#test-environment-during-the-training)
		- [Test environment's keyboard shortcut](#test-environments-keyboard-shortcut)
	- [Work achievement](#work-achievement)
		- [About ICIRA](#about-icira)
		- [About RAS](#about-ras)

## Handbook about our work  
This is a handbook about our work's difference with Nvidia's work.
1. We provide a test platform that you can view the rewards, obs or any information you want to know during the training process. What's more, it can be used to record offline datas in Simulate environment to apply the offline algorithm without real environment.
   
   It provides you more choices to train your tasks with low cost.
2. We will provide a multi-agent train api with multi-ppo, multi-sac algorithms to help you train your multi-agent tasks. 
### Installation

1. The essential package [rl_games](https://github.com/Denys88/rl_games) use the newest version v1.3.2, have to download the package manually.

   Copy to path `/IsaacGymEnvs/issacgymenvs/rl_games`

2. Replace `isaacgymenvs` folder in `/IsaacGymEnvs/isaacgymenvs`


### Test environment during the training

1. Place `test_new.py` under the same folder with `train.py`, should be `/IsaacGymEnvs/isaacgymenvs`

2. Check the test config file in `test_config.ini`

3. Running in the environment

   ```
   python test.py
   ```

4. Save pose data under folder `./test_save`, hdf5 under `./test_save/hdf5`

### Test environment's keyboard shortcut

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
| `V`                     | Check task stage in certain obs                              |
| `9`                     | Pause tracking                                               |
| `L`                     | Override Franka DOFs                                         |
| `UP, DOWN, LEFT, RIGHT` | Drive franka position manually in `xz` plane                 |
| `[`, `]`                | Drive franka position manually in `y` axis                   |
| `LEFT_SHIFT`            | Switch driving between left/right Franka                     |


## Work achievement
We try to show you our efforts from two papers, the ICIRA one is accepted, as soon as the RAS one is still working!
### About ICIRA
**Title: 'Mixline: A Hybrid Reinforcement Learning Framework for Long-horizon Bimanual Coffee Stirring Task'**

The camera-ready version:
> https://mycuhk-my.sharepoint.com/:b:/g/personal/1155169521_link_cuhk_edu_hk/ETAQRHX-UcRJr2E4gKzROZgBYWgVX_5-VrLBOjP8WaIq0w?e=l3X3hg

The powerpoint about our ICIRA work:
>https://mycuhk-my.sharepoint.com/:p:/g/personal/1155169521_link_cuhk_edu_hk/EaD-nU8Dc2hHreP4dB2QQ-oB5EnMBSCWO4AFzSCv7wUfpQ?e=D1nHnh

Mixline diagram:
![Image](https://pic4.zhimg.com/80/v2-81285504d720391134c6a857056e33bc.png)
### About RAS
The paper for RAS is still working.

