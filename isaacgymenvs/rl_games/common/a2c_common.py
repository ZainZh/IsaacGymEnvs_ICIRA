import os

from rl_games.common import tr_helpers
from rl_games.common import vecenv
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch.moving_mean_std import MovingMeanStd
from rl_games.algos_torch.self_play_manager import SelfPlayManager
from rl_games.algos_torch import torch_ext
from rl_games.common import schedulers
from rl_games.common.experience import ExperienceBuffer
from rl_games.common.interval_summary_writer import IntervalSummaryWriter
from rl_games.common.diagnostics import DefaultDiagnostics, PpoDiagnostics
from rl_games.algos_torch import model_builder
from rl_games.interfaces.base_algorithm import BaseAlgorithm
import numpy as np
import time
import gym

from datetime import datetime
from tensorboardX import SummaryWriter
import torch
from torch import nn

from time import sleep

from rl_games.common import common_losses


def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


# 45-755 A2CBase
class A2CBase(BaseAlgorithm):
    def __init__(self, base_name, params):
        self.config = config = params['config']
        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'

        # This helps in PBT when we need to restart an experiment with the exact same name, rather than
        # generating a new name with the timestamp every time.
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
            self.experiment_name_left = full_experiment_name + '_left'
            self.experiment_name_right = full_experiment_name + '_right'
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")

        self.config = config
        self.add_cql = config.get('add_cql', False)
        self.cql_version = self.config.get('cql_version', 0)
        self.algo_observer = config['features']['observer']
        self.algo_observer.before_init(base_name, config, self.experiment_name)

        self.load_networks(params)
        self.multi_gpu = config.get('multi_gpu', False)
        self.rank = 0
        self.rank_size = 1
        self.curr_frames = 0
        self.multi_franka = self.config.get('multi_franka', False)
        if self.multi_gpu:
            from rl_games.distributed.hvd_wrapper import HorovodWrapper
            self.hvd = HorovodWrapper()
            self.config = self.hvd.update_algo_config(config)
            self.rank = self.hvd.rank
            self.rank_size = self.hvd.rank_size

        self.use_diagnostics = config.get('use_diagnostics', False)

        if self.use_diagnostics and self.rank == 0:
            self.diagnostics = PpoDiagnostics()
        else:
            self.diagnostics = DefaultDiagnostics()

        self.network_path = config.get('network_path', "./nn/")
        self.log_path = config.get('log_path', "runs/")
        self.env_config = config.get('env_config', {})
        self.num_actors = config['num_actors']
        self.env_name = config['env_name']
        self.vec_env = None

        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            if self.multi_franka:
                self.env_info = self.vec_env.get_env_info()
            else:
                self.env_info = self.vec_env.get_env_info()

        self.ppo_device = config.get('device', self.vec_env.env.device_id)  # or cuda:0?
        print('Env info:')
        print(self.env_info)
        self.value_size = self.env_info.get('value_size', 1)
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.use_action_masks = config.get('use_action_masks', False)  # False
        self.is_train = config.get('is_train', True)

        self.central_value_config = self.config.get('central_value_config', None)
        self.has_central_value = self.central_value_config is not None  # False
        self.truncate_grads = self.config.get('truncate_grads', False)

        if self.has_central_value:
            self.state_space = self.env_info.get('state_space', None)
            if isinstance(self.state_space, gym.spaces.Dict):
                self.state_shape = {}
                for k, v in self.state_space.spaces.items():
                    self.state_shape[k] = v.shape
            else:
                self.state_shape = self.state_space.shape

        self.self_play_config = self.config.get('self_play_config', None)
        self.has_self_play_config = self.self_play_config is not None

        self.self_play = config.get('self_play', False)
        self.save_freq = config.get('save_frequency', 0)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.ppo = config.get('ppo', True)
        self.max_epochs = self.config.get('max_epochs', 1e6)

        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.linear_lr = config['lr_schedule'] == 'linear'

        if self.is_adaptive_lr:
            self.kl_threshold = config['kl_threshold']
            self.scheduler = schedulers.AdaptiveScheduler(self.kl_threshold)
        elif self.linear_lr:
            self.scheduler = schedulers.LinearScheduler(float(config['learning_rate']),
                                                        max_steps=self.max_epochs,
                                                        apply_to_entropy=config.get('schedule_entropy', False),
                                                        start_entropy_coef=config.get('entropy_coef'))
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.horizon_length = config['horizon_length']
        self.seq_len = self.config.get('seq_length', 4)
        self.bptt_len = self.config.get('bptt_length', self.seq_len)
        self.normalize_advantage = config['normalize_advantage']
        self.normalize_rms_advantage = config.get('normalize_rms_advantage', False)  # False
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)  # True
        self.truncate_grads = self.config.get('truncate_grads', False)
        self.has_phasic_policy_gradients = False

        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape

        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']

        self.games_to_track = self.config.get('games_to_track', 100)
        print('current training device:', self.ppo_device)
        self.game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)

        self.obs = None
        self.obs_left = None
        self.obs_right = None
        # Todo: add two new obs
        self.games_num = self.config['minibatch_size'] // self.seq_len  # it is used only for current rnn implementation
        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors
        assert (('minibatch_size_per_env' in self.config) or ('minibatch_size' in self.config))
        self.minibatch_size_per_env = self.config.get('minibatch_size_per_env', 0)
        self.minibatch_size = self.config.get('minibatch_size', self.num_actors * self.minibatch_size_per_env)
        self.mini_epochs_num = self.config['mini_epochs']
        self.num_minibatches = self.batch_size // self.minibatch_size
        assert (self.batch_size % self.minibatch_size == 0)

        self.mixed_precision = self.config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # add multi_franka param

        self.last_lr = self.config['learning_rate']
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -100500
        self.play_time = 0
        self.epoch_num = 0
        self.curr_frames = 0
        # allows us to specify a folder where all experiments will reside
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.entropy_coef = self.config['entropy_coef']

        if self.rank == 0:
            writer = SummaryWriter(self.summaries_dir)
            if self.population_based_training:
                self.writer = IntervalSummaryWriter(writer, self.config)
            else:
                self.writer = writer
        else:
            self.writer = None

        self.value_bootstrap = self.config.get('value_bootstrap')

        self.use_smooth_clamp = self.config.get('use_smooth_clamp', False)

        if self.use_smooth_clamp:
            self.actor_loss_func = common_losses.smoothed_actor_loss
        else:
            self.actor_loss_func = common_losses.actor_loss

        if self.normalize_advantage and self.normalize_rms_advantage:
            momentum = self.config.get('adv_rms_momentum', 0.5)  # '0.25'
            self.advantage_mean_std = MovingMeanStd((1,), momentum=momentum).to(self.ppo_device)

        self.is_tensor_obses = False

        self.last_rnn_indices = None
        self.last_state_indices = None

        # self_play
        if self.has_self_play_config:
            print('Initializing SelfPlay Manager')
            self.self_play_manager = SelfPlayManager(self.self_play_config, self.writer)

        # features
        self.algo_observer = config['features']['observer']

        self.soft_aug = config['features'].get('soft_augmentation', None)
        self.has_soft_aug = self.soft_aug is not None
        # soft augmentation not yet supported
        assert not self.has_soft_aug

    def trancate_gradients_and_step(self):
        if self.multi_gpu:
            self.optimizer.synchronize()

            # self.truncate_grads is True
        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        if self.multi_gpu:
            with self.optimizer.skip_synchronize():
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)
        has_central_value_net = self.config.get('central_value_config') is not None
        if has_central_value_net:
            print('Adding Central Value Network')
            if 'model' not in params['config']['central_value_config']:
                params['config']['central_value_config']['model'] = {'name': 'central_value'}
            network = builder.load(params['config']['central_value_config'])
            self.config['central_value_config']['network'] = network

    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses_left, c_losses_left,
                    entropies_left, kls_left,
                    last_lr_left, lr_mul_left, a_losses_right, c_losses_right, entropies_right, kls_right,
                    last_lr_right, lr_mul_right, frame, scaled_time, scaled_play_time, curr_frames):
        # do we need scaled time?
        self.diagnostics.send_info(self.writer)
        self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer.add_scalar('performance/rl_update_time', update_time, frame)
        self.writer.add_scalar('performance/step_inference_time', play_time, frame)
        self.writer.add_scalar('performance/step_time', step_time, frame)

        self.writer.add_scalar('losses/a_loss_left', torch_ext.mean_list(a_losses_left).item(), frame)
        self.writer.add_scalar('losses/c_loss_left', torch_ext.mean_list(c_losses_left).item(), frame)
        self.writer.add_scalar('losses/entropy_left', torch_ext.mean_list(entropies_left).item(), frame)
        self.writer.add_scalar('info/last_lr_left', last_lr_left * lr_mul_left, frame)
        self.writer.add_scalar('info/lr_mul_left', lr_mul_left, frame)
        self.writer.add_scalar('info/e_clip_left', self.e_clip * lr_mul_left, frame)
        self.writer.add_scalar('info/kl_left', torch_ext.mean_list(kls_left).item(), frame)

        self.writer.add_scalar('losses/a_loss_right', torch_ext.mean_list(a_losses_right).item(), frame)
        self.writer.add_scalar('losses/c_loss_right', torch_ext.mean_list(c_losses_right).item(), frame)
        self.writer.add_scalar('losses/entropy_right', torch_ext.mean_list(entropies_right).item(), frame)
        self.writer.add_scalar('info/last_lr_right', last_lr_right * lr_mul_right, frame)
        self.writer.add_scalar('info/lr_mul_right', lr_mul_right, frame)
        self.writer.add_scalar('info/e_clip_right', self.e_clip * lr_mul_right, frame)
        self.writer.add_scalar('info/kl_right', torch_ext.mean_list(kls_right).item(), frame)
        self.writer.add_scalar('info/epochs_right', epoch_num, frame)
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    def set_eval(self):
        self.model.eval()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.train()

    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr])
            self.hvd.broadcast_value(lr_tensor, 'learning_rate')
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'rnn_states': self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states': states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            if self.has_central_value:
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states': states,
                    'actions': None,
                    'is_done': self.dones,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model.eval()
                processed_obs = self._preproc_obs(obs['obs'])
                input_dict = {
                    'is_train': False,
                    'prev_actions': None,
                    'obs': processed_obs,
                    'rnn_states': self.rnn_states
                }
                result = self.model(input_dict)
                value = result['values']
            return value

    @property
    def device(self):
        return self.ppo_device

    def reset_envs(self):
        self.obs = self.env_reset()

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors': self.num_actors,
            'horizon_length': self.horizon_length,
            'has_central_value': self.has_central_value,
            'use_action_masks': self.use_action_masks
        }
        self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)

        val_shape = (self.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

            total_agents = self.num_agents * self.num_actors
            num_seqs = self.horizon_length // self.seq_len
            assert ((self.horizon_length * total_agents // self.num_minibatches) % self.seq_len == 0)
            self.mb_rnn_states = [
                torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype=torch.float32,
                            device=self.ppo_device) for s in self.rnn_states]

    def init_rnn_from_model(self, model):
        self.is_rnn = self.model.is_rnn()

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert (self.observation_space.dtype != np.int8)
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.ppo_device)
            else:
                obs = torch.FloatTensor(obs).to(self.ppo_device)
        return obs

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:
            upd_obs = {'obs': upd_obs}
        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(
                dones).to(self.ppo_device), infos

    def env_reset(self):
        obs = self.vec_env.reset()
        obs = self.obs_to_tensors(obs)
        return obs

    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t + 1]
                nextvalues = mb_extrinsic_values[t + 1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs

    def discount_values_masks(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards,
                              mb_masks):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t + 1]
                nextvalues = mb_extrinsic_values[t + 1]
            nextnonterminal = nextnonterminal.unsqueeze(1)
            masks_t = mb_masks[t].unsqueeze(1)
            delta = (mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t])
            mb_advs[t] = lastgaelam = (delta + self.gamma * self.tau * nextnonterminal * lastgaelam) * masks_t
        return mb_advs

    def clear_stats(self):
        batch_size = self.num_agents * self.num_actors
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -100500
        self.algo_observer.after_clear_stats()

    def update_epoch(self):
        pass

    def train(self):
        pass

    def prepare_dataset(self, batch_dict):
        pass

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

    def train_epoch_multi(self):
        self.vec_env.set_train_info(self.frame, self)

    def train_actor_critic(self, obs_dict, opt_step=True):
        pass

    def calc_gradients(self):
        pass

    def get_central_value(self, obs_dict):
        return self.central_value_net.get_value(obs_dict)

    def train_central_value(self):
        return self.central_value_net.train_net()

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = self.epoch_num
        state['optimizer'] = self.optimizer.state_dict()
        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()
        state['frame'] = self.frame

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state['last_mean_rewards'] = self.last_mean_rewards

        if self.vec_env is not None:
            env_state = self.vec_env.get_env_state()
            state['env_state'] = env_state

        return state

    def set_full_state_weights(self, weights):
        self.set_weights(weights)
        self.epoch_num = weights['epoch']
        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])
        self.optimizer.load_state_dict(weights['optimizer'])
        self.frame = weights.get('frame', 0)
        self.last_mean_rewards = weights.get('last_mean_rewards', -100500)

        env_state = weights.get('env_state', None)

        if self.vec_env is not None:
            self.vec_env.set_env_state(env_state)

    def get_weights(self):
        state = self.get_stats_weights()
        state['model'] = self.model.state_dict()
        return state

    def get_stats_weights(self, model_stats=False):
        state = {}
        if self.mixed_precision:
            state['scaler'] = self.scaler.state_dict()
        if self.has_central_value:
            state['central_val_stats'] = self.central_value_net.get_stats_weights(model_stats)
        if model_stats:
            if self.normalize_input:
                state['running_mean_std'] = self.model.running_mean_std.state_dict()
            if self.normalize_value:
                state['reward_mean_std'] = self.model.value_mean_std.state_dict()

        return state

    def set_stats_weights(self, weights):
        if self.normalize_rms_advantage:
            self.advantage_mean_std.load_state_dic(weights['advantage_mean_std'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])
        if self.normalize_value and 'normalize_value' in weights:
            self.model.value_mean_std.load_state_dict(weights['reward_mean_std'])
        if self.mixed_precision and 'scaler' in weights:
            self.scaler.load_state_dict(weights['scaler'])

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        self.set_stats_weights(weights)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(
                    1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = self.dones.view(self.num_actors, self.num_agents).all(dim=1).nonzero(as_tuple=False)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict

    def play_steps_rnn(self):
        # update_list = self.update_list
        mb_rnn_states = self.mb_rnn_states
        step_time = 0.0

        for n in range(self.horizon_length):
            if n % self.seq_len == 0:
                for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                    mb_s[n // self.seq_len, :, :, :] = s

            if self.has_central_value:
                self.central_value_net.pre_step_rnn(n)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.rnn_states = res_dict['rnn_states']
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones.byte())

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(
                    1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = self.dones.view(self.num_actors, self.num_agents).all(dim=1).nonzero(as_tuple=False)
            if len(all_done_indices) > 0:
                for s in self.rnn_states:
                    s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
                if self.has_central_value:
                    self.central_value_net.post_step_rnn(all_done_indices)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()

        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        states = []
        for mb_s in mb_rnn_states:
            t_size = mb_s.size()[0] * mb_s.size()[2]
            h_size = mb_s.size()[3]
            states.append(mb_s.permute(1, 2, 0, 3).reshape(-1, t_size, h_size))
        batch_dict['rnn_states'] = states
        batch_dict['step_time'] = step_time
        return batch_dict


# 755-984 Discrete
class DiscreteA2CBase(A2CBase):
    def __init__(self, base_name, params):
        A2CBase.__init__(self, base_name, params)
        batch_size = self.num_agents * self.num_actors
        action_space = self.env_info['action_space']
        if type(action_space) is gym.spaces.Discrete:
            self.actions_shape = (self.horizon_length, batch_size)
            self.actions_num = action_space.n
            self.is_multi_discrete = False
        if type(action_space) is gym.spaces.Tuple:
            self.actions_shape = (self.horizon_length, batch_size, len(action_space))
            self.actions_num = [action.n for action in action_space]
            self.is_multi_discrete = True
        self.is_discrete = True

    def init_tensors(self):
        A2CBase.init_tensors(self)
        self.update_list = ['actions', 'neglogpacs', 'values']
        if self.use_action_masks:
            self.update_list += ['action_masks']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']

    def train_epoch(self):
        super().train_epoch()

        self.set_eval()
        play_time_start = time.time()

        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        self.set_train()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        a_losses = []
        c_losses = []
        entropies = []
        kls = []

        if self.has_central_value:
            self.train_central_value()

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                av_kls = self.hvd.average_value(av_kls, 'ep_kls')

            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,
                                                                    av_kls.item())
            self.update_lr(self.last_lr)
            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval()  # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict[
                   'step_time'], play_time, update_time, total_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul

    def prepare_dataset(self, batch_dict):
        rnn_masks = batch_dict.get('rnn_masks', None)

        returns = batch_dict['returns']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        dones = batch_dict['dones']
        rnn_states = batch_dict.get('rnn_states', None)
        advantages = returns - values

        obses = batch_dict['obses']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks

        if self.use_action_masks:
            dataset_dict['action_masks'] = batch_dict['action_masks']

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['dones'] = dones
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def train(self):
        self.init_tensors()
        self.mean_rewards = self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        # self.frame = 0  # loading from checkpoint
        self.obs = self.env_reset()

        if self.multi_gpu:
            self.hvd.setup_algo(self)

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()

            if self.multi_gpu:
                self.hvd.sync_stats(self)

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            total_time += sum_time
            curr_frames = self.curr_frames
            self.frame += curr_frames
            should_exit = False
            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time

                frame = self.frame // self.num_agents

                if self.print_stats:
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(
                        f'fps step: {fps_step:.1f} fps step and policy inference: {fps_step_inference:.1f} fps total: {fps_total:.1f} epoch: {epoch_num}/{self.max_epochs}')

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses,
                                 entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    # removed equal signs (i.e. "rew=") from the checkpoint name since it messes with hydra CLI parsing
                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))
                        if self.last_mean_rewards > self.config['score_to_win']:
                            print('Network won!')
                            self.save(os.path.join(self.nn_dir, checkpoint_name))
                            should_exit = True

                if epoch_num > self.max_epochs:
                    self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))
                    print('MAX EPOCHS NUM!')
                    should_exit = True
                update_time = 0
            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit).float()
                self.hvd.broadcast_value(should_exit_t, 'should_exit')
                should_exit = should_exit_t.bool().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num


### 1197_continuous
class ContinuousA2CBase(A2CBase):
    def __init__(self, base_name, params):
        A2CBase.__init__(self, base_name, params)
        self.is_discrete = False
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        self.bounds_loss_coef = self.config.get('bounds_loss_coef', None)

        self.clip_actions = self.config.get('clip_actions', True)

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)

    def preprocess_actions(self, actions):
        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions

    def init_tensors(self):
        A2CBase.init_tensors(self)
        self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']

    def train_epoch(self):
        super().train_epoch()

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(
                    self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)

            av_kls = torch_ext.mean_list(ep_kls)

            if self.multi_gpu:
                av_kls = self.hvd.average_value(av_kls, 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,
                                                                    av_kls.item())
            self.update_lr(self.last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval()  # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict[
                   'step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = returns - values

        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['dones'] = dones
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            self.hvd.setup_algo(self)

        while True:
            epoch_num = self.update_epoch()
            print('\033[1;32m---------------- Epoch {} ----------------\033[0m'.format(epoch_num))
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            if self.multi_gpu:
                self.hvd.sync_stats(self)
            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False
            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames
                self.frame += curr_frames
                if self.print_stats:
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(
                        f'fps step: {fps_step:.1f} fps step and policy inference: {fps_step_inference:.1f}  fps total: {fps_total:.1f}')

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses,
                                 entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        print('mean_rewards: {}'.format(mean_rewards[i]))
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards[0] <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))
                        if self.last_mean_rewards > self.config['score_to_win']:
                            print('Network won!')
                            self.save(os.path.join(self.nn_dir, checkpoint_name))
                            should_exit = True

                if epoch_num > self.max_epochs:
                    self.save(os.path.join(self.nn_dir,
                                           'last_' + self.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(
                                               mean_rewards)))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                update_time = 0
            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit).float()
                self.hvd.broadcast_value(should_exit_t, 'should_exit')
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num


# 1571
class ContinuousMultiA2CBase(A2CBase):

    # ToDo: add split Datasets function

    def __init__(self, base_name, params):
        import h5py
        A2CBase.__init__(self, base_name, params)
        self.is_discrete = False
        action_space = self.env_info['action_space']
        config = self.config
        self.actions_num = action_space.shape[0] * 2
        self.bounds_loss_coef = self.config.get('bounds_loss_coef', None)

        self.clip_actions = self.config.get('clip_actions', True)

        self.rnn_states_left = None
        self.rnn_states_right = None

        # add new algo_observer
        self.algo_observer_left = self.config['features']['observer']
        self.algo_observer_left.before_init(base_name, self.config, self.experiment_name_left)
        self.algo_observer_right = self.config['features']['observer']
        self.algo_observer_right.before_init(base_name, self.config, self.experiment_name_right)

        if self.is_adaptive_lr:
            self.scheduler_left = schedulers.AdaptiveScheduler(self.kl_threshold)
            self.scheduler_right = schedulers.AdaptiveScheduler(self.kl_threshold)

        elif self.linear_lr:
            self.scheduler_left = schedulers.LinearScheduler(float(config['learning_rate']),
                                                             max_steps=self.max_epochs,
                                                             apply_to_entropy=config.get('schedule_entropy', False),
                                                             start_entropy_coef=config.get('entropy_coef'))
            self.scheduler_right = schedulers.LinearScheduler(float(config['learning_rate']),
                                                              max_steps=self.max_epochs,
                                                              apply_to_entropy=config.get('schedule_entropy', False),
                                                              start_entropy_coef=config.get('entropy_coef'))
        else:
            self.scheduler_left = schedulers.IdentityScheduler()
            self.scheduler_right = schedulers.IdentityScheduler()

        # add two games rewards and lengths
        self.game_rewards_left = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_lengths_left = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        self.game_rewards_right = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_lengths_right = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)

        self.scaler_left = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.scaler_right = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.last_lr_left = self.config['learning_rate']
        self.last_lr_right = self.config['learning_rate']
        self.offlinePPO=self.config['offline_ppo']
        self.curr_frames_left = 0
        self.curr_frames_right = 0
        self.entropy_coef_left = self.config['entropy_coef']
        self.entropy_coef_right = self.config['entropy_coef']

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)
        self.actions_low = torch.cat((self.actions_low, self.actions_low), 0)
        self.actions_high = torch.cat((self.actions_high, self.actions_high), 0)


        date_file = h5py.File('./replay_buffer/replay_buff512000.hdf5', 'r')

        # left action
        self.data_actions_left = torch.tensor(np.array(date_file['actions_left']), dtype=torch.float,
                                              device=self.device)
        self.data_actions_left = self.data_actions_left[0:512000, :]

        self.data_actions_left = torch.reshape(self.data_actions_left, (250, 2048, 9))

        # left obs
        self.data_next_obs_left = torch.tensor(np.array(date_file['next_observations_left']), dtype=torch.float,
                                               device=self.device)
        self.data_next_obs_left = self.data_next_obs_left[0:512000, :]

        self.data_next_obs_left = torch.reshape(self.data_next_obs_left, (250, 2048, 37))

        # right action
        self.data_actions_right = torch.tensor(np.array(date_file['actions_right']), dtype=torch.float,
                                               device=self.device)

        self.data_actions_right = self.data_actions_right[0:512000, :]

        self.data_actions_right = torch.reshape(self.data_actions_right, (250, 2048, 9))

        # right obs
        self.data_next_obs_right = torch.tensor(np.array(date_file['next_observations_right']), dtype=torch.float,
                                                device=self.device)

        self.data_next_obs_right = self.data_next_obs_right[0:512000, :]

        self.data_next_obs_right = torch.reshape(self.data_next_obs_right, (250, 2048, 37))

    def env_reset_multi(self):
        obs_left, obs_right = self.vec_env.reset_multi()
        obs_left = self.obs_to_tensors(obs_left)
        obs_right = self.obs_to_tensors(obs_right)
        return obs_left, obs_right

    def preprocess_actions(self, actions):

        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors': self.num_actors,
            'horizon_length': self.horizon_length,
            'has_central_value': self.has_central_value,
            'use_action_masks': self.use_action_masks
        }
        self.experience_buffer_left = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)
        self.experience_buffer_right = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)

        current_rewards_shape = (batch_size, self.value_size)
        # Todo add two current_rewards and dones
        self.current_rewards_left = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths_left = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.current_rewards_right = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths_right = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)

        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)
        self.dones_spoon = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)
        self.dones_cup = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

        if self.is_rnn_left:
            self.rnn_states_left = self.model_left.get_default_rnn_state()
            self.rnn_states_left = [s.to(self.ppo_device) for s in self.rnn_states_left]

            total_agents = self.num_agents * self.num_actors
            num_seqs = self.horizon_length // self.seq_len
            assert ((self.horizon_length * total_agents // self.num_minibatches) % self.seq_len == 0)
            self.mb_rnn_states_left = [
                torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype=torch.float32,
                            device=self.ppo_device) for s in self.rnn_states_left]
        if self.is_rnn_right:
            self.rnn_states_right = self.model_right.get_default_rnn_state()
            self.rnn_states_right = [s.to(self.ppo_device) for s in self.rnn_states_right]

            total_agents = self.num_agents * self.num_actors
            num_seqs = self.horizon_length // self.seq_len
            assert ((self.horizon_length * total_agents // self.num_minibatches) % self.seq_len == 0)
            self.mb_rnn_states_right = [
                torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype=torch.float32,
                            device=self.ppo_device) for s in self.rnn_states_right]

        self.update_list_left = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list_left = self.update_list_left + ['obses', 'states', 'dones']
        self.update_list_right = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list_right = self.update_list_right + ['obses', 'states', 'dones']

    # Todo: split and combine Franka actions
    def action_split(self, actions):
        actions_left = actions[:, 0:9]
        actions_right = actions[:, 9:18]
        return actions_left, actions_right

    def action_combine(self, actions_left, actions_right):
        actions = torch.cat((actions_left, actions_right), 1)
        return actions

    def trancate_gradients_and_step_left(self):
        if self.truncate_grads:
            self.scaler_left.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model_left.parameters(), self.grad_norm)

        self.scaler_left.step(self.optimizer)
        self.scaler_left.update()

    def trancate_gradients_and_step_right(self):
        if self.truncate_grads:
            self.scaler_right.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model_right.parameters(), self.grad_norm)

        self.scaler_right.step(self.optimizer)
        self.scaler_right.update()

    def get_action_values_left(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model_left.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'rnn_states': self.rnn_states_left
        }

        with torch.no_grad():
            res_dict = self.model_left(input_dict)
        return res_dict

    def get_action_values_right(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model_right.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'rnn_states': self.rnn_states_right
        }

        with torch.no_grad():
            res_dict = self.model_right(input_dict)
        return res_dict

    def get_values_left(self, obs):
        with torch.no_grad():
            if self.has_central_value:
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states': states,
                    'actions': None,
                    'is_done': self.dones_spoon,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model_left.eval()
                processed_obs = self._preproc_obs(obs['obs'])
                input_dict = {
                    'is_train': False,
                    'prev_actions': None,
                    'obs': processed_obs,
                    'rnn_states': self.rnn_states_left
                }
                result = self.model_left(input_dict)
                value = result['values']
            return value

    def get_values_right(self, obs):
        with torch.no_grad():
            if self.has_central_value:
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states': states,
                    'actions': None,
                    'is_done': self.dones_cup,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model_right.eval()
                processed_obs = self._preproc_obs(obs['obs'])
                input_dict = {
                    'is_train': False,
                    'prev_actions': None,
                    'obs': processed_obs,
                    'rnn_states': self.rnn_states_right
                }
                result = self.model_right(input_dict)
                value = result['values']
            return value

    def update_lr_left(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr])
            self.hvd.broadcast_value(lr_tensor, 'learning_rate')
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def update_lr_right(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr])
            self.hvd.broadcast_value(lr_tensor, 'learning_rate')
            lr = lr_tensor.item()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def init_rnn_from_model_left(self, model):
        self.is_rnn_left = self.model_left.is_rnn()

    def init_rnn_from_model_right(self, model):
        self.is_rnn_right = self.model_right.is_rnn()

    def set_train_multi(self):
        self.model_left.train()
        self.model_right.train()

    def set_eval_multi(self):
        self.model_left.eval()
        self.model_right.eval()

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs_left, obs_right, rewards, dones, dones_spoon, dones_cup, infos, left_reward, right_reward = self.vec_env.step_multi(
            actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
                left_reward = left_reward.unsqueeze(1)
                right_reward = right_reward.unsqueeze(1)

            return self.obs_to_tensors(obs_left), self.obs_to_tensors(obs_right), rewards.to(self.ppo_device), \
                   dones.to(self.ppo_device), dones_spoon.to(self.ppo_device), dones_cup.to(self.ppo_device), infos, \
                   left_reward.to(self.ppo_device), right_reward.to(self.ppo_device)

        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
                left_reward = np.expand_dims(left_reward, axis=1)
                right_reward = np.expand_dims(right_reward, axis=1)

            return self.obs_to_tensors(obs_left), self.obs_to_tensors(obs_right), rewards.to(self.ppo_device), dones.to(
                self.ppo_device), infos, \
                   left_reward.to(self.ppo_device), right_reward.to(self.ppo_device)

    def clear_stats(self):
        self.game_rewards_left.clear_left()
        self.game_lengths_left.clear_left()
        self.game_rewards_right.clear_right()
        self.game_lengths_right.clear_right()
        self.algo_observer_left.after_clear_stats()
        self.algo_observer_right.after_clear_stats()
        # Todo: need more consideration
        self.mean_rewards = self.last_mean_rewards = -100500

    def get_full_state_weights(self):
        state_left, state_right = self.get_weights()
        state_left['epoch'] = self.epoch_num
        state_right['epoch'] = self.epoch_num
        state_left['optimizer'] = self.optimizer.state_dict()
        state_right['optimizer'] = self.optimizer.state_dict()
        if self.has_central_value:
            state_left['assymetric_vf_nets'] = self.central_value_net.state_dict()
            state_right['assymetric_vf_nets'] = self.central_value_net.state_dict()
        state_left['frame'] = self.frame
        state_right['frame'] = self.frame

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state_left['last_mean_rewards'] = self.last_mean_rewards
        state_right['last_mean_rewards'] = self.last_mean_rewards

        if self.vec_env is not None:
            env_state = self.vec_env.get_env_state()
            state_left['env_state'] = env_state
            state_right['env_state'] = env_state

        return state_left, state_right

    def set_full_state_weights(self, weights):
        self.set_weights(weights)
        self.epoch_num = weights['epoch']
        self.optimizer.load_state_dict(weights['optimizer'])

        self.frame = weights.get('frame', 0)
        self.last_mean_rewards = weights.get('last_mean_rewards', -100500)

        env_state = weights.get('env_state', None)

        if self.vec_env is not None:
            self.vec_env.set_env_state(env_state)

    def get_weights(self):
        state_left, state_right = self.get_stats_weights()
        state_left['model'] = self.model_left.state_dict()
        state_right['model'] = self.model_right.state_dict()
        return state_left, state_right

    def get_stats_weights(self, model_stats=False):
        state_left = {}
        state_right = {}
        if self.mixed_precision:
            state_left['scaler'] = self.scaler_left.state_dict()
            state_right['scaler'] = self.scaler_right.state_dict()
        if self.has_central_value:
            state_left['central_val_stats'] = self.central_value_net.get_stats_weights(model_stats)
            state_right['central_val_stats'] = self.central_value_net.get_stats_weights(model_stats)
        if model_stats:
            if self.normalize_input:
                state_left['running_mean_std'] = self.model_left.running_mean_std.state_dict()
                state_right['running_mean_std'] = self.model_right.running_mean_std.state_dict()
            if self.normalize_value:
                state_left['reward_mean_std'] = self.model_left.value_mean_std.state_dict()
                state_right['reward_mean_std'] = self.model_right.value_mean_std.state_dict()

        return state_left, state_right

    def set_stats_weights(self, weights):
        if self.normalize_input and 'running_mean_std' in weights:
            self.model_left.running_mean_std.load_state_dict(weights['running_mean_std'])
            self.model_right.running_mean_std.load_state_dict(weights['running_mean_std'])
        if self.normalize_value and 'normalize_value' in weights:
            self.model_left.value_mean_std.load_state_dict(weights['reward_mean_std'])
            self.model_right.value_mean_std.load_state_dict(weights['reward_mean_std'])
        if self.mixed_precision and 'scaler' in weights:
            self.scaler_left.load_state_dict(weights['scaler'])
            self.scaler_right.load_state_dict(weights['scaler'])

    def set_weights(self, weights):
        self.model_left.load_state_dict(weights['model_left'])
        self.model_right.load_state_dict(weights['model_right'])
        self.set_stats_weights(weights)

    def play_steps_rnn_multi(self):
        # update_list = self.update_list
        update_list_left = self.update_list_left
        update_list_right = self.update_list_right
        mb_rnn_states_left = self.mb_rnn_states_left
        mb_rnn_states_right = self.mb_rnn_states_right
        step_time = 0.0

        for n in range(self.horizon_length):
            if n % self.seq_len == 0:
                for s, mb_s in zip(self.rnn_states_left, mb_rnn_states_left):
                    mb_s[n // self.seq_len, :, :, :] = s
                for s, mb_s in zip(self.rnn_states_right, mb_rnn_states_right):
                    mb_s[n // self.seq_len, :, :, :] = s

            if self.has_central_value:
                self.central_value_net.pre_step_rnn(n)

            res_dict_left = self.get_action_values_left(self.obs_left)
            res_dict_right = self.get_action_values_right(self.obs_right)

            self.rnn_states_left = res_dict_left['rnn_states']
            self.rnn_states_right = res_dict_right['rnn_states']
            self.experience_buffer_left.update_data_left('obses', n, self.obs_left['obs'])
            self.experience_buffer_left.update_data_left('dones', n, self.dones_spoon.byte())
            self.experience_buffer_right.update_data_right('obses', n, self.obs_right['obs'])
            self.experience_buffer_right.update_data_right('dones', n, self.dones_cup.byte())

            for k in update_list_left:
                self.experience_buffer_left.update_data_left(k, n, res_dict_left[k])
            for k in update_list_right:
                self.experience_buffer_right.update_data_right(k, n, res_dict_right[k])

            step_time_start = time.time()
            # actions_new = self.obs['obs'][:,-18:]
            # Todo: add another franka arm actions
            actions_new = self.action_combine(res_dict_left['actions'], res_dict_right['actions'])
            self.obs_left, self.obs_right, rewards, self.dones, self.dones_spoon, self.dones_cup, infos, rewards_left, rewards_right = self.env_step(
                actions_new)
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            # two reward

            shaped_rewards_left = self.rewards_shaper(rewards_left)
            shaped_rewards_right = self.rewards_shaper(rewards_right)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards_left += self.gamma * res_dict_left['values'] * self.cast_obs(
                    infos['time_outs']).unsqueeze(
                    1).float()
                shaped_rewards_right += self.gamma * res_dict_right['values'] * self.cast_obs(
                    infos['time_outs']).unsqueeze(
                    1).float()

            self.experience_buffer_left.update_data_left('rewards', n, shaped_rewards_left)
            self.experience_buffer_right.update_data_right('rewards', n, shaped_rewards_right)

            self.current_rewards_left += rewards_left
            self.current_rewards_right += rewards_right
            self.current_lengths_left += 1
            self.current_lengths_right += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            all_done_indices_left = self.dones_spoon.nonzero(as_tuple=False)
            all_done_indices_right = self.dones_cup.nonzero(as_tuple=False)
            env_done_indices = self.dones.view(self.num_actors, self.num_agents).all(dim=1).nonzero(as_tuple=False)
            env_done_indices_left = self.dones_spoon.view(self.num_actors, self.num_agents).all(dim=1).nonzero(
                as_tuple=False)
            env_done_indices_right = self.dones_cup.view(self.num_actors, self.num_agents).all(dim=1).nonzero(
                as_tuple=False)
            if len(all_done_indices_left) > 0:
                for s in self.rnn_states_left:
                    s[:, all_done_indices_left, :] = s[:, all_done_indices_left, :] * 0.0
            if len(all_done_indices_right) > 0:
                for s in self.rnn_states_right:
                    s[:, all_done_indices_right, :] = s[:, all_done_indices_right, :] * 0.0

            self.game_rewards_left.update_left(self.current_rewards_left[env_done_indices_left])
            self.game_rewards_right.update_right(self.current_rewards_right[env_done_indices_right])
            self.game_lengths_left.update_left(self.current_lengths_left[env_done_indices_left])
            self.game_lengths_right.update_right(self.current_lengths_right[env_done_indices_right])
            self.algo_observer_left.process_infos(infos, env_done_indices_left)
            self.algo_observer_right.process_infos(infos, env_done_indices_right)

            not_dones_left = 1.0 - self.dones_spoon.float()
            not_dones_right = 1.0 - self.dones_cup.float()

            self.current_rewards_left = self.current_rewards_left * not_dones_left.unsqueeze(1)
            self.current_rewards_right = self.current_rewards_right * not_dones_right.unsqueeze(1)
            self.current_lengths_left = self.current_lengths_left * not_dones_left
            self.current_lengths_right = self.current_lengths_right * not_dones_right

        last_values_left = self.get_values_left(self.obs_left)
        last_values_right = self.get_values_right(self.obs_right)

        fdones_left = self.dones_spoon.float()
        fdones_right = self.dones_cup.float()
        mb_fdones_left = self.experience_buffer_left.tensor_dict_left['dones'].float()
        mb_fdones_right = self.experience_buffer_right.tensor_dict_right['dones'].float()

        mb_values_left = self.experience_buffer_left.tensor_dict_left['values']
        mb_values_right = self.experience_buffer_right.tensor_dict_right['values']
        mb_rewards_left = self.experience_buffer_left.tensor_dict_left['rewards']
        mb_rewards_right = self.experience_buffer_right.tensor_dict_right['rewards']
        mb_advs_left = self.discount_values(fdones_left, last_values_left, mb_fdones_left, mb_values_left,
                                            mb_rewards_left)
        mb_advs_right = self.discount_values(fdones_right, last_values_right, mb_fdones_right, mb_values_right,
                                             mb_rewards_right)
        mb_returns_left = mb_advs_left + mb_values_left
        mb_returns_right = mb_advs_right + mb_values_right
        batch_dict_left = self.experience_buffer_left.get_transformed_list_left(swap_and_flatten01,
                                                                                self.tensor_list_left)
        batch_dict_right = self.experience_buffer_right.get_transformed_list_right(swap_and_flatten01,
                                                                                   self.tensor_list_right)
        batch_dict_left['returns'] = swap_and_flatten01(mb_returns_left)
        batch_dict_right['returns'] = swap_and_flatten01(mb_returns_right)
        batch_dict_left['played_frames'] = self.batch_size
        batch_dict_right['played_frames'] = self.batch_size
        states_left = []
        states_right = []
        for mb_s in mb_rnn_states_left:
            t_size = mb_s.size()[0] * mb_s.size()[2]
            h_size = mb_s.size()[3]
            states_left.append(mb_s.permute(1, 2, 0, 3).reshape(-1, t_size, h_size))
        for mb_s in mb_rnn_states_right:
            t_size = mb_s.size()[0] * mb_s.size()[2]
            h_size = mb_s.size()[3]
            states_right.append(mb_s.permute(1, 2, 0, 3).reshape(-1, t_size, h_size))
        batch_dict_left['rnn_states'] = states_left
        batch_dict_right['rnn_states'] = states_right
        batch_dict_left['step_time'] = step_time
        batch_dict_right['step_time'] = step_time
        return batch_dict_left, batch_dict_right

    def play_steps_multi(self):
        # Todo: add lists
        update_list_left = self.update_list_left
        update_list_right = self.update_list_right
        step_time = 0.0

        for n in range(self.horizon_length):

            res_dict_left = self.get_action_values_left(self.obs_left)
            res_dict_right = self.get_action_values_right(self.obs_right)

            # self.experience_buffer.update_data('obses', n, self.obs['obs'])
            # self.experience_buffer.update_data('dones', n, self.dones)

            # Todo: add informations
            self.experience_buffer_left.update_data_left('obses', n, self.obs_left['obs'])
            self.experience_buffer_left.update_data_left('dones', n, self.dones_spoon)
            self.experience_buffer_right.update_data_right('obses', n, self.obs_right['obs'])
            self.experience_buffer_right.update_data_right('dones', n, self.dones_cup)

            # Todo: add informations
            for k in update_list_left:
                self.experience_buffer_left.update_data_left(k, n, res_dict_left[k])
            if self.has_central_value:
                self.experience_buffer_left.update_data_left('states', n, self.obs_left['states'])

            for k in update_list_right:
                self.experience_buffer_right.update_data_right(k, n, res_dict_right[k])
            if self.has_central_value:
                self.experience_buffer_right.update_data_right('states', n, self.obs_right['states'])

            step_time_start = time.time()

            # split actions from two dics and combine actions_left in dic1 with actions_right in dic2

            # actions_new = self.obs['obs'][:,-18:]
            actions_new = self.action_combine(res_dict_left['actions'], res_dict_right['actions'])
            # Todo: add another franka arm actions
            self.obs_left, self.obs_right, rewards, self.dones, self.dones_spoon, self.dones_cup, infos, rewards_left, rewards_right = self.env_step(
                actions_new)

            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)
            # two rewards

            shaped_rewards_left = self.rewards_shaper(rewards_left)
            shaped_rewards_right = self.rewards_shaper(rewards_right)

            # if self.value_bootstrap and 'time_outs' in infos:
            #     shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(
            #         1).float()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards_left += self.gamma * res_dict_left['values'] * self.cast_obs(
                    infos['time_outs']).unsqueeze(
                    1).float()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards_right += self.gamma * res_dict_right['values'] * self.cast_obs(
                    infos['time_outs']).unsqueeze(
                    1).float()

            # self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer_left.update_data_left('rewards', n, shaped_rewards_left)
            self.experience_buffer_right.update_data_right('rewards', n, shaped_rewards_right)

            # self.current_rewards += rewards
            # self.current_lengths += 1
            self.current_rewards_left += rewards_left
            self.current_lengths_left += 1
            self.current_rewards_right += rewards_right
            self.current_lengths_right += 1

            # shaped_rewards = self.rewards_shaper(rewards)

            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = self.dones.view(self.num_actors, self.num_agents).all(dim=1).nonzero(as_tuple=False)
            env_done_indices_left = self.dones_spoon.view(self.num_actors, self.num_agents).all(dim=1).nonzero(
                as_tuple=False)
            env_done_indices_right = self.dones_cup.view(self.num_actors, self.num_agents).all(dim=1).nonzero(
                as_tuple=False)

            # self.game_rewards.update(self.current_rewards[env_done_indices])
            # self.game_lengths.update(self.current_lengths[env_done_indices])
            # self.algo_observer.process_infos(infos, env_done_indices)

            self.game_rewards_left.update_left(self.current_rewards_left[env_done_indices_left])
            self.game_lengths_left.update_left(self.current_lengths_left[env_done_indices_left])
            self.game_rewards_right.update_right(self.current_rewards_right[env_done_indices_right])
            self.game_lengths_right.update_right(self.current_lengths_right[env_done_indices_right])
            self.algo_observer_left.process_infos(infos, env_done_indices_left)
            self.algo_observer_right.process_infos(infos, env_done_indices_right)
            not_dones = 1.0 - self.dones.float()
            not_dones_left = 1.0 - self.dones_spoon.float()
            not_dones_right = 1.0 - self.dones_cup.float()
            self.current_rewards_left = self.current_rewards_left * not_dones_left.unsqueeze(1)
            self.current_lengths_left = self.current_lengths_left * not_dones_left
            self.current_rewards_right = self.current_rewards_right * not_dones_right.unsqueeze(1)
            self.current_lengths_right = self.current_lengths_right * not_dones_right

        last_values_left = self.get_values_left(self.obs_left)
        last_values_right = self.get_values_right(self.obs_right)
        fdones = self.dones.float()
        fdones_left = self.dones_spoon.float()
        fdones_right = self.dones_cup.float()
        mb_fdones_left = self.experience_buffer_left.tensor_dict_left['dones'].float()
        mb_values_left = self.experience_buffer_left.tensor_dict_left['values']
        mb_rewards_left = self.experience_buffer_left.tensor_dict_left['rewards']
        mb_fdones_right = self.experience_buffer_right.tensor_dict_right['dones'].float()
        mb_values_right = self.experience_buffer_right.tensor_dict_right['values']
        mb_rewards_right = self.experience_buffer_right.tensor_dict_right['rewards']
        mb_advs_left = self.discount_values(fdones_left, last_values_left, mb_fdones_left, mb_values_left,
                                            mb_rewards_left)
        mb_returns_left = mb_advs_left + mb_values_left
        mb_advs_right = self.discount_values(fdones_right, last_values_right, mb_fdones_right, mb_values_right,
                                             mb_rewards_right)
        mb_returns_right = mb_advs_right + mb_values_right

        # Todo: add other batch_dict for another arm
        batch_dict_left = self.experience_buffer_left.get_transformed_list_left(swap_and_flatten01,
                                                                                self.tensor_list_left)
        batch_dict_left['returns'] = swap_and_flatten01(mb_returns_left)
        batch_dict_left['played_frames'] = self.batch_size
        batch_dict_left['step_time'] = step_time

        batch_dict_right = self.experience_buffer_right.get_transformed_list_right(swap_and_flatten01,
                                                                                   self.tensor_list_right)
        batch_dict_right['returns'] = swap_and_flatten01(mb_returns_right)
        batch_dict_right['played_frames'] = self.batch_size
        batch_dict_right['step_time'] = step_time

        return batch_dict_left, batch_dict_right

    def train_epoch_multi(self):
        super().train_epoch_multi()
        self.set_eval_multi()
        play_time_start = time.time()
        with torch.no_grad():
            # self.is_rnn is False
            if self.is_rnn_left and self.is_rnn_right:  # Todo rewrite play_steps_rnn_multi
                batch_dict_left, batch_dict_right = self.play_steps_rnn_multi_offline()  # evaluion: the interaction
            else:
                batch_dict_left, batch_dict_right = self.play_steps_multi()

        play_time_end = time.time()
        update_time_start = time.time()
        # rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train_multi()
        self.curr_frames_left = batch_dict_left.pop('played_frames')
        self.curr_frames_right = batch_dict_right.pop('played_frames')
        self.prepare_dataset_left(batch_dict_left)
        self.prepare_dataset_right(batch_dict_right)
        self.algo_observer_left.after_steps()
        self.algo_observer_right.after_steps()
        if self.has_central_value:
            self.train_central_value()
        # Todo: init another franka's losses
        a_losses_left = []
        c_losses_left = []
        b_losses_left = []
        entropies_left = []
        kls_left = []
        a_losses_right = []
        c_losses_right = []
        b_losses_right = []
        entropies_right = []
        kls_right = []
        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls_left = []
            ep_kls_right = []
            for i in range(len(self.dataset_left)):

                a_loss_left, c_loss_left, entropy_left, kl_left, last_lr_left, lr_mul_left, cmu_left, csigma_left, b_loss_left, \
                a_loss_right, c_loss_right, entropy_right, kl_right, last_lr_right, lr_mul_right, cmu_right, csigma_right, b_loss_right = self.train_actor_critic_multi(
                    self.dataset_left[i], self.dataset_right[i], self.data_actions_left[i], self.data_next_obs_left[i],
                    self.data_actions_right[i], self.data_next_obs_right[i])

                a_losses_left.append(a_loss_left)
                c_losses_left.append(c_loss_left)
                ep_kls_left.append(kl_left)
                entropies_left.append(entropy_left)
                if self.bounds_loss_coef is not None:
                    b_losses_left.append(b_loss_left)

                a_losses_right.append(a_loss_right)
                c_losses_right.append(c_loss_right)
                ep_kls_right.append(kl_right)
                entropies_right.append(entropy_right)
                if self.bounds_loss_coef is not None:
                    b_losses_right.append(b_loss_right)
                self.dataset_left.update_mu_sigma(cmu_left, csigma_left)
                self.dataset_right.update_mu_sigma(cmu_right, csigma_right)

            av_kls_left = torch_ext.mean_list(ep_kls_left)
            av_kls_right = torch_ext.mean_list(ep_kls_right)

            if self.multi_gpu:
                av_kls_left = self.hvd.average_value(av_kls_left, 'ep_kls')
                av_kls_right = self.hvd.average_value(av_kls_right, 'ep_kls')
            self.last_lr_left, self.entropy_coef_left = self.scheduler_left.update(self.last_lr_left,
                                                                                   self.entropy_coef_left,
                                                                                   self.epoch_num, 0,
                                                                                   av_kls_left.item())
            self.last_lr_right, self.entropy_coef_right = self.scheduler_right.update(self.last_lr_right,
                                                                                      self.entropy_coef_right,
                                                                                      self.epoch_num, 0,
                                                                                      av_kls_right.item())
            self.update_lr_left(self.last_lr_left)
            self.update_lr_right(self.last_lr_right)

            kls_left.append(av_kls_left)
            kls_right.append(av_kls_right)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model_left.running_mean_std.eval()  # don't need to update statstics more than one miniepoch
                self.model_right.running_mean_std.eval()  # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        # Todo: return two arm's loss and other params.

        return batch_dict_left[
                   'step_time'], play_time, update_time, total_time, \
               a_losses_left, c_losses_left, b_losses_left, entropies_left, kls_left, last_lr_left, lr_mul_left, \
               a_losses_right, c_losses_right, b_losses_right, entropies_right, kls_right, last_lr_right, lr_mul_right

    def prepare_dataset_left(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']

        mus = batch_dict['mus']

        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = returns - values

        if self.normalize_value:
            self.value_mean_std_left.train()
            values = self.value_mean_std_left(values)
            returns = self.value_mean_std_left(returns)
            self.value_mean_std_left.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn_left:

                advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset_left.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['dones'] = dones
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def prepare_dataset_right(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']

        mus = batch_dict['mus']

        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = returns - values

        if self.normalize_value:
            self.value_mean_std_right.train()
            values = self.value_mean_std_right(values)
            returns = self.value_mean_std_right(returns)
            self.value_mean_std_right.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn_right:
                advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset_right.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['dones'] = dones
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def train(self):
        # global mean_reward_left, mean_length_left, mean_reward_right, mean_length_right
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs_left, self.obs_right = self.env_reset_multi()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            self.hvd.setup_algo(self)

        while True:
            epoch_num = self.update_epoch()
            print('\033[1;32m---------------- Epoch {} ----------------\033[0m'.format(epoch_num))
            # Todo: add other train epoch
            step_time, play_time, update_time, sum_time, \
            a_losses_left, c_losses_left, b_losses_left, entropies_left, kls_left, last_lr_left, lr_mul_left, \
            a_losses_right, c_losses_right, b_losses_right, entropies_right, kls_right, last_lr_right, lr_mul_right = self.train_epoch_multi()

            total_time += sum_time
            frame = self.frame // self.num_agents

            if self.multi_gpu:
                self.hvd.sync_stats(self)
            # cleaning memory to optimize space

            self.dataset_left.update_values_dict(None)
            self.dataset_right.update_values_dict(None)
            should_exit = False

            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames
                self.frame += curr_frames
                if self.print_stats:
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(
                        f'fps step: {fps_step:.1f} fps step and policy inference: {fps_step_inference:.1f} fps total: {fps_total:.1f} epoch: {epoch_num}/{self.max_epochs}')

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses_left, c_losses_left,
                                 entropies_left, kls_left, last_lr_left, lr_mul_left, a_losses_right, c_losses_right,
                                 entropies_right, kls_right, last_lr_right, lr_mul_right, frame, scaled_time,
                                 scaled_play_time, curr_frames)
                if len(b_losses_left) > 0:
                    self.writer.add_scalar('losses/bounds_loss_left', torch_ext.mean_list(b_losses_left).item(), frame)
                if len(b_losses_right) > 0:
                    self.writer.add_scalar('losses/bounds_loss_right', torch_ext.mean_list(b_losses_right).item(),
                                           frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                if self.game_rewards_left.current_size_left > 0:
                    mean_reward_left = self.game_rewards_left.get_mean_left()
                    mean_length_left = self.game_lengths_left.get_mean_left()
                    self.mean_rewards_left = mean_reward_left[0]
                    print(
                        'mean_rewards_left: {}, mean_length_left: {}'.format(self.mean_rewards_left, mean_length_left))
                if self.game_rewards_right.current_size_right > 0:
                    mean_reward_right = self.game_rewards_right.get_mean_right()
                    mean_length_right = self.game_lengths_right.get_mean_right()
                    self.mean_rewards_right = mean_reward_right[0]
                    # print('current length: {}'.format(self.current_lengths))
                    # print('current rewards: {}'.format(self.current_rewards / self.current_lengths)
                    print('mean_rewards_right: {}, mean_length_right: {}'.format(self.mean_rewards_right,
                                                                                 mean_length_right))

                    for i in range(self.value_size):
                        rewards_name = 'left_rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_reward_left[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_reward_left[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_reward_left[i], total_time)
                        rewards_name = 'right_rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_reward_right[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_reward_right[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_reward_right[i], total_time)

                    self.writer.add_scalar('episode_lengths_left/step', mean_length_left, frame)
                    self.writer.add_scalar('episode_lengths_left/iter', mean_length_left, epoch_num)
                    self.writer.add_scalar('episode_lengths_left/time', mean_length_left, total_time)
                    self.writer.add_scalar('episode_lengths_right/step', mean_length_right, frame)
                    self.writer.add_scalar('episode_lengths_right/iter', mean_length_right, epoch_num)
                    self.writer.add_scalar('episode_lengths_right/time', mean_length_right, total_time)
                    self.writer.add_scalar('episode_lengths_all/step', mean_length_left + mean_length_right, frame)
                    self.writer.add_scalar('episode_lengths_all/iter', mean_length_left + mean_length_right, epoch_num)
                    self.writer.add_scalar('episode_lengths_allo/time', mean_length_left + mean_length_right,
                                           total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(
                        mean_reward_left[0] + mean_reward_right[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (
                                mean_reward_left[0] + mean_reward_right[0] <= self.last_mean_rewards):
                            self.save_multi(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_reward_left[0] + mean_reward_right[
                        0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best left rewards: ', mean_reward_left[0])
                        print('saving next best right rewards: ', mean_reward_right[0])
                        self.last_mean_rewards = mean_reward_left[0] + mean_reward_right[0]
                        self.save_multi(os.path.join(self.nn_dir, self.config['name']))
                        if self.last_mean_rewards > self.config['score_to_win']:
                            print('Network won!')
                            self.save_multi(os.path.join(self.nn_dir, checkpoint_name))
                            should_exit = True

                if epoch_num > self.max_epochs:
                    self.save_multi(os.path.join(self.nn_dir,
                                                 'last_' + self.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(
                                                     mean_reward_left + mean_reward_right)))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                update_time = 0
            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit).float()
                self.hvd.broadcast_value(should_exit_t, 'should_exit')
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num
