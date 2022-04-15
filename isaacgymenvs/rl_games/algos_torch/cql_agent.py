from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import vecenv
from rl_games.common import schedulers
from rl_games.common import experience
from rl_games.interfaces.base_algorithm import BaseAlgorithm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from rl_games.algos_torch import model_builder
from torch import optim
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
import os


# copied from SACAgent
class CQLAgent(BaseAlgorithm):
    def __init__(self, base_name, params):
        self.config = config = params['config']
        print('----------------------------------')
        print(config)
        print('----------------------------------')
        # TODO: Get obs shape and self.network
        self.load_networks(params)
        self.base_init(base_name, config)
        self.num_seed_steps = config["num_seed_steps"]  # random explore
        self.gamma = config["gamma"]
        self.critic_tau = config["critic_tau"]
        self.batch_size = config["batch_size"]
        self.init_alpha = config["init_alpha"]
        self.learnable_temperature = config["learnable_temperature"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.replay_buffer_path = config["replay_buffer_path"]
        self.num_steps_per_episode = config.get("num_steps_per_episode", 1000)
        self.normalize_input = config.get("normalize_input", False)

        self.max_env_steps = config.get("max_env_steps", 1000)  # temporary, in future we will use other approach

        print('explore_steps: {}, batch_size: {}, num_actor: {}, num_agent: {}'.format(self.num_seed_steps,
                                                                                       self.batch_size, self.num_actors,
                                                                                       self.num_agents))

        self.num_frames_per_epoch = self.num_actors * self.num_steps_per_episode

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).float().to(self.sac_device)
        self.log_alpha.requires_grad = True
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        net_config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'normalize_input': self.normalize_input,
            'normalize_input': self.normalize_input,
        }
        self.model = self.network.build(net_config)
        self.model.to(self.sac_device)

        self.actor_optimizer = torch.optim.Adam(self.model.sac_network.actor.parameters(),
                                                lr=self.config['actor_lr'],
                                                betas=self.config.get("actor_betas", [0.9, 0.999]))

        self.critic_optimizer = torch.optim.Adam(self.model.sac_network.critic.parameters(),
                                                 lr=self.config["critic_lr"],
                                                 betas=self.config.get("critic_betas", [0.9, 0.999]))

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.config["alpha_lr"],
                                                    betas=self.config.get("alphas_betas", [0.9, 0.999]))
        # CQL(L)  alpha_prime
        self.with_lagrange = config['with_lagrange']
        if self.with_lagrange:
            self.target_action_gap = config['lagrange_thresh']
            self.log_alpha_prime = torch.zeros(1, requires_grad=True, device=self.sac_device)
            self.alpha_prime_optimizer = torch.optim.Adam([self.log_alpha_prime],
                                                          lr=self.config["critic_lr"],
                                                          betas=self.config.get("critic_betas", [0.9, 0.999]))

        self.replay_buffer = experience.VectorizedReplayBuffer(self.env_info['observation_space'].shape,
                                                               self.env_info['action_space'].shape,
                                                               self.replay_buffer_size,
                                                               self.sac_device)

        self.target_entropy_coef = config.get("target_entropy_coef", 0.5)
        self.target_entropy = self.target_entropy_coef * -self.env_info['action_space'].shape[0]
        print("Target entropy", self.target_entropy)
        self.step = 0
        self.algo_observer = config['features']['observer']

        # TODO: Is there a better way to get the maximum number of episodes?
        self.max_episodes = torch.ones(self.num_actors, device=self.sac_device) * self.num_steps_per_episode
        # self.episode_lengths = np.zeros(self.num_actors, dtype=int)

        # add CQL init args
        self.num_random = config['num_random']
        self.min_q_version = config['min_q_version']
        self.with_lagrange = config['with_lagrange']
        self.temp = 1.0
        self.min_q_weight = config['min_q_weight']

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def base_init(self, base_name, config):
        self.env_config = config.get('env_config', {})
        self.num_actors = config.get('num_actors', 1)
        self.env_name = config['env_name']
        print("Env name:", self.env_name)

        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)  # ('rlgpu',10,{})
            self.env_info = self.vec_env.get_env_info()

        self.sac_device = config.get('device', 'cuda:0')
        self.ppo_device = self.sac_device
        # temporary:
        print('Env info: {}'.format(self.env_info))

        # self.rewards_shaper = config['reward_shaper']
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        # self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.c_loss = nn.MSELoss()
        # self.c2_loss = nn.SmoothL1Loss()

        self.save_best_after = config.get('save_best_after', 500)
        print('save_best_after: {}'.format(self.save_best_after))
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.max_epochs = self.config.get('max_epochs', 1e6)

        self.network = config['network']  # build in load_networks
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.obs_shape = self.observation_space.shape

        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self.sac_device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.sac_device)
        self.obs = None

        self.min_alpha = torch.tensor(np.log(1)).float().to(self.sac_device)

        self.frame = 0
        self.update_time = 0
        self.last_mean_rewards = -100500
        self.play_time = 0
        self.epoch_num = 0

        # allows us to specify a folder where all experiments will reside
        self.train_dir = config.get('train_dir', 'runs')
        # a folder inside of train_dir containing everything related to a particular experiment
        file_time = datetime.now().strftime("%m%d-%H-%M-%S")
        self.experiment_name = config.get('name')
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.nn_dir, file_time)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.writer = SummaryWriter(self.experiment_dir + '/summaries/' + file_time)
        print("Run Directory:", self.experiment_dir + '/summaries/' + file_time)

        self.is_tensor_obses = None
        self.is_rnn = False
        self.last_rnn_indices = None
        self.last_state_indices = None

    def init_tensors(self):
        if self.observation_space.dtype == np.uint8:
            torch_dtype = torch.uint8
        else:
            torch_dtype = torch.float32
        batch_size = self.num_agents * self.num_actors

        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self.sac_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.sac_device)

        self.dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self.sac_device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def device(self):
        return self.sac_device

    def get_full_state_weights(self):
        state = self.get_weights()

        state['steps'] = self.step
        state['actor_optimizer'] = self.actor_optimizer.state_dict()
        state['critic_optimizer'] = self.critic_optimizer.state_dict()
        state['log_alpha_optimizer'] = self.log_alpha_optimizer.state_dict()
        if self.with_lagrange:
            state['alpha_prime_optimizer'] = self.alpha_prime_optimizer.state_dict()

        return state

    def get_weights(self):
        state = {'actor': self.model.sac_network.actor.state_dict(),
                 'critic': self.model.sac_network.critic.state_dict(),
                 'critic_target': self.model.sac_network.critic_target.state_dict()}
        return state

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def set_weights(self, weights):
        self.model.sac_network.actor.load_state_dict(weights['actor'])
        self.model.sac_network.critic.load_state_dict(weights['critic'])
        self.model.sac_network.critic_target.load_state_dict(weights['critic_target'])

        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])

    def set_full_state_weights(self, weights):
        self.set_weights(weights)

        self.step = weights['steps']
        self.actor_optimizer.load_state_dict(weights['actor_optimizer'])
        self.critic_optimizer.load_state_dict(weights['critic_optimizer'])
        self.log_alpha_optimizer.load_state_dict(weights['log_alpha_optimizer'])
        if self.with_lagrange:
            self.alpha_prime_optimizer.load_state_dict(weights['alpha_prime_optimizer'])

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn, map_location=self.device)
        self.set_full_state_weights(checkpoint)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()

    def update_critic(self, obs, action, reward, next_obs, not_done, step):
        with torch.no_grad():
            dist = self.model.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.model.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * log_prob

            target_Q = reward + (not_done * self.gamma * target_V)
            target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.model.critic(obs, action)

        critic1_loss = self.c_loss(current_Q1, target_Q)
        critic2_loss = self.c_loss(current_Q2, target_Q)
        # critic_loss = critic1_loss + critic2_loss 

        # add CQL here
        random_actions_tensor = torch.FloatTensor(current_Q2.shape[0] *
                                                  self.num_random, action.shape[-1]).uniform_(-1, 1).to(self.sac_device)
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random,
                                                                     network=self.model.actor)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.num_random,
                                                                        network=self.model.actor)
        q1_rand, q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.model.critic)
        q1_curr_actions, q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.model.critic)
        q1_next_actions, q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor,
                                                                   network=self.model.critic)

        cat_q1 = torch.cat(
            [q1_rand, current_Q1.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, current_Q2.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
        )
        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)

        if self.min_q_version == 3:
            # importance sammpled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(),
                 q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(),
                 q2_curr_actions - curr_log_pis.detach()], 1
            )

        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1, ).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1, ).mean() * self.min_q_weight * self.temp

        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - current_Q1.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - current_Q2.mean() * self.min_q_weight

        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)  # target_action_gap=threshold
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            alpha_prime_loss = torch.zeros_like(std_q2, requires_grad=False)

        critic1_loss = critic1_loss + min_qf1_loss
        critic2_loss = critic2_loss + min_qf2_loss

        # CQL add end
        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.detach(), critic1_loss.detach(), critic2_loss.detach(), min_qf1_loss.detach(), \
               min_qf2_loss.detach(), std_q1.detach(), std_q2.detach(), alpha_prime_loss.detach()

    def update_actor_and_alpha(self, obs, step):
        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = False

        dist = self.model.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True).mean()
        actor_Q1, actor_Q2 = self.model.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (torch.max(self.alpha.detach(), self.min_alpha) * log_prob - actor_Q)
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = True

        if self.learnable_temperature:
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            self.log_alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        else:
            alpha_loss = None

        return actor_loss.detach(), entropy.detach(), self.alpha.detach(), alpha_loss  # TODO: maybe not self.alpha

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    def update(self, step):
        obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)
        not_done = ~done

        obs = self.preproc_obs(obs)
        next_obs = self.preproc_obs(next_obs)
        # add return value
        critic_loss, critic1_loss, critic2_loss, min_qf1_loss, min_qf2_loss, std_q1, std_q2, alpha_prime_loss \
            = self.update_critic(obs, action, reward, next_obs, not_done, step)

        actor_loss, entropy, alpha, alpha_loss = self.update_actor_and_alpha(obs, step)

        actor_loss_info = actor_loss, entropy, alpha, alpha_loss
        self.soft_update_params(self.model.sac_network.critic, self.model.sac_network.critic_target,
                                self.critic_tau)
        return actor_loss_info, critic1_loss, critic2_loss, min_qf1_loss, min_qf2_loss, std_q1, std_q2, alpha_prime_loss

    def preproc_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs['obs']
        return obs

    def env_step(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = self.vec_env.step(actions)  # (obs_space) -> (n, obs_space)

        self.step += self.num_actors
        if self.is_tensor_obses:
            return obs, rewards, dones, infos
        else:
            return torch.from_numpy(obs).to(self.sac_device), torch.from_numpy(rewards).to(
                self.sac_device), torch.from_numpy(dones).to(self.sac_device), infos

    def env_reset(self):
        with torch.no_grad():
            obs = self.vec_env.reset()
        obs = self.preproc_obs(obs)

        # obs_to_tensors() in 1.1.3, changed
        if self.is_tensor_obses is None:
            self.is_tensor_obses = torch.is_tensor(obs)
            print("Observations are tensors:", self.is_tensor_obses)

        if self.is_tensor_obses:
            return obs.to(self.sac_device)
        else:
            return torch.from_numpy(obs).to(self.sac_device)

    def act(self, obs, action_dim, sample=False):
        obs = self.preproc_obs(obs)
        dist = self.model.actor(obs)
        actions = dist.sample() if sample else dist.mean
        actions = actions.clamp(*self.action_range)
        assert actions.ndim == 2
        return actions

    def extract_actor_stats(self, actor_losses, entropies, alphas, alpha_losses, actor_loss_info):
        actor_loss, entropy, alpha, alpha_loss = actor_loss_info

        actor_losses.append(actor_loss)
        entropies.append(entropy)
        if alpha_losses is not None:
            alphas.append(alpha)
            alpha_losses.append(alpha_loss)

    def clear_stats(self):
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -100500
        self.algo_observer.after_clear_stats()

    def play_steps(self, random_exploration=False):
        total_time_start = time.time()
        total_update_time = 0
        step_time = 0.0
        actor_losses = []
        entropies = []
        alphas = []
        alpha_losses = []
        critic1_losses = []
        critic2_losses = []
        # add cql params
        min_qf1_losses = []
        min_qf2_losses = []
        std_q1s = []
        std_q2s = []
        alpha_prime_losses = []

        obs = self.obs
        for _ in range(self.num_steps_per_episode):
            self.set_eval()
            if random_exploration:
                action = torch.rand((self.num_actors, *self.env_info["action_space"].shape),
                                    device=self.sac_device) * 2 - 1
            else:
                with torch.no_grad():
                    action = self.act(obs.float(), self.env_info["action_space"].shape, sample=True)

            step_start = time.time()
            with torch.no_grad():
                next_obs, rewards, dones, infos = self.env_step(action)
            step_end = time.time()

            self.current_rewards += rewards
            self.current_lengths += 1

            step_time += (step_end - step_start)

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])  # update game rewards if have done in envs
            self.game_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - dones.float()

            self.algo_observer.process_infos(infos, done_indices)  # Log infos for envs which are done

            no_timeouts = self.current_lengths != self.max_env_steps
            dones = dones * no_timeouts

            self.current_rewards = self.current_rewards * not_dones  # if done, reset the corresponding current_rewards
            self.current_lengths = self.current_lengths * not_dones  # if done, reset the corresponding current_lengths

            if isinstance(obs, dict):
                obs = obs['obs']
            if isinstance(next_obs, dict):
                next_obs = next_obs['obs']

            rewards = self.rewards_shaper(rewards)  # scale_value

            self.replay_buffer.add(obs, action, torch.unsqueeze(rewards, 1), next_obs, torch.unsqueeze(dones, 1))

            self.obs = obs = next_obs.clone()

            if not random_exploration:
                self.set_train()
                update_time_start = time.time()
                actor_loss_info, critic1_loss, critic2_loss, min_qf1_loss, min_qf2_loss, std_q1, std_q2, alpha_prime_loss = self.update(
                    self.epoch_num)
                update_time_end = time.time()
                update_time = update_time_end - update_time_start

                self.extract_actor_stats(actor_losses, entropies, alphas, alpha_losses, actor_loss_info)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)
                min_qf1_losses.append(min_qf1_loss)
                min_qf2_losses.append(min_qf2_loss)
                std_q1s.append(std_q1)
                std_q2s.append(std_q2)
                alpha_prime_losses.append(alpha_prime_loss)
            else:
                update_time = 0

            total_update_time += update_time

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        play_time = total_time - total_update_time

        return step_time, play_time, total_update_time, total_time, actor_losses, entropies, alphas, alpha_losses, \
               critic1_losses, critic2_losses, min_qf1_losses, min_qf2_losses, std_q1s, std_q2s, alpha_prime_losses

    def train_epoch(self):
        if self.epoch_num < self.num_seed_steps and not self.config['load_checkpoint']:  # Random explore
            step_time, play_time, total_update_time, total_time, actor_losses, entropies, alphas, alpha_losses, \
            critic1_losses, critic2_losses, min_qf1_losses, min_qf2_losses, std_q1s, std_q2s, alpha_prime_losses \
                = self.play_steps(random_exploration=True)
        else:  # RL training
            step_time, play_time, total_update_time, total_time, actor_losses, entropies, alphas, alpha_losses, \
            critic1_losses, critic2_losses, min_qf1_losses, min_qf2_losses, std_q1s, std_q2s, alpha_prime_losses \
                = self.play_steps(random_exploration=False)

        return step_time, play_time, total_update_time, total_time, actor_losses, entropies, alphas, alpha_losses, \
               critic1_losses, critic2_losses, min_qf1_losses, min_qf2_losses, std_q1s, std_q2s, alpha_prime_losses

    def train(self):
        self.init_tensors()
        self.algo_observer.after_init(self)
        self.last_mean_rewards = -100500
        total_time = 0
        # rep_count = 0
        self.frame = 0
        self.obs = self.env_reset()
        print('\033[1;33mStart training\033[0m')  # add hint

        while True:
            self.epoch_num += 1
            print('\033[1;32m---------------- Epoch {} ----------------\033[0m'.format(self.epoch_num))

            step_time, play_time, update_time, epoch_total_time, actor_losses, entropies, alphas, alpha_losses, \
            critic1_losses, critic2_losses, min_qf1_losses, min_qf2_losses, std_q1s, std_q2s, alpha_prime_losses \
                = self.train_epoch()

            total_time += epoch_total_time
            scaled_time = epoch_total_time
            scaled_play_time = play_time
            curr_frames = self.num_frames_per_epoch
            self.frame += curr_frames
            frame = self.frame  # TODO: Fix frame

            if self.print_stats:
                fps_step = curr_frames / scaled_play_time
                fps_total = curr_frames / scaled_time
                print(f'fps step: {fps_step:.1f} fps total: {fps_total:.1f}')

            self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
            self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
            self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
            self.writer.add_scalar('performance/rl_update_time', update_time, frame)
            self.writer.add_scalar('performance/step_inference_time', play_time, frame)
            self.writer.add_scalar('performance/step_time', step_time, frame)

            if self.epoch_num >= self.num_seed_steps or self.config['load_checkpoint']:
                self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(actor_losses).item(), frame)
                self.writer.add_scalar('losses/c1_loss', torch_ext.mean_list(critic1_losses).item(), frame)
                self.writer.add_scalar('losses/c2_loss', torch_ext.mean_list(critic2_losses).item(), frame)
                # std Q value add
                self.writer.add_scalar('losses/std_c1_loss', torch_ext.mean_list(std_q1s).item(), frame)
                self.writer.add_scalar('losses/std_c2_loss', torch_ext.mean_list(std_q2s).item(), frame)
                self.writer.add_scalar('losses/min_c1_loss', torch_ext.mean_list(min_qf1_losses).item(), frame)
                self.writer.add_scalar('losses/min_c2_loss', torch_ext.mean_list(min_qf2_losses).item(), frame)
                if self.with_lagrange:
                    self.writer.add_scalar('losses/alpha_prime_loss', torch_ext.mean_list(alpha_prime_losses).item(),
                                           frame)
                # end cql
                self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), frame)
                if alpha_losses[0] is not None:
                    self.writer.add_scalar('losses/alpha_loss', torch_ext.mean_list(alpha_losses).item(), frame)
                self.writer.add_scalar('info/alpha', torch_ext.mean_list(alphas).item(), frame)

            self.writer.add_scalar('info/epochs', self.epoch_num, frame)
            self.algo_observer.after_print_stats(frame, self.epoch_num, total_time)

            if self.game_rewards.current_size > 0:
                mean_rewards = self.game_rewards.get_mean()
                mean_lengths = self.game_lengths.get_mean()

                print('current length: {}'.format(self.current_lengths))
                print('current rewards: {}'.format(self.current_rewards / self.current_lengths))
                print('mean_rewards: {}, mean_length: {}'.format(mean_rewards, mean_lengths))

                self.writer.add_scalar('rewards/step', mean_rewards, frame)
                self.writer.add_scalar('rewards/iter', mean_rewards, self.epoch_num)
                self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                self.writer.add_scalar('episode_lengths/iter', mean_lengths, self.epoch_num)
                self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                # <editor-fold desc="Checkpoint">
                if mean_rewards > self.last_mean_rewards and self.epoch_num >= self.save_best_after:
                    print('saving next best rewards: ', mean_rewards)
                    self.last_mean_rewards = mean_rewards
                    self.save(
                        os.path.join(self.checkpoint_dir, 'ep_' + str(self.epoch_num) + '_rew_' + str(mean_rewards)))
                    # if self.last_mean_rewards > self.config.get('score_to_win', float('inf')):  #
                    #     print('Network won!')
                    #     self.save(os.path.join(self.checkpoint_dir,
                    #                            'won_ep=' + str(self.epoch_num) + '_rew=' + str(mean_rewards)))
                    #     return self.last_mean_rewards, self.epoch_num

                if self.epoch_num > self.max_epochs:
                    self.save(os.path.join(self.checkpoint_dir,
                                           'last_ep_' + str(self.epoch_num) + '_rew_' + str(mean_rewards)))
                    print('MAX EPOCHS NUM!')
                    return self.last_mean_rewards, self.epoch_num
                update_time = 0

                if self.epoch_num % 100 == 0:
                    self.save(
                        os.path.join(self.checkpoint_dir, 'ep_' + str(self.epoch_num) + '_rew_' + str(mean_rewards)))
                    print('model backup save')
                # </editor-fold>

    # copy from CQL
    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        dist = network(obs_temp)
        new_obs_actions = dist.rsample()
        new_obs_log_pi = dist.log_prob(new_obs_actions).sum(-1, keepdim=True)

        return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        # change, q1,q2=self.model.critic(obs,action)
        preds_q1, preds_q2 = network(obs_temp, actions)
        preds_q1 = preds_q1.view(obs.shape[0], num_repeat, 1)
        preds_q2 = preds_q2.view(obs.shape[0], num_repeat, 1)
        return preds_q1, preds_q2

    def load_hdf5(self, dataset_path):
        import h5py
        _dataset = h5py.File(dataset_path, 'r')
        _obs = torch.tensor(np.array(_dataset['observations']), dtype=torch.float, device=self.device)
        _actions = torch.tensor(np.array(_dataset['actions']), dtype=torch.float, device=self.device)
        _rewards = torch.tensor(np.array(_dataset['rewards']), dtype=torch.float, device=self.device)
        _next_obs = torch.tensor(np.array(_dataset['next_observations']), dtype=torch.float, device=self.device)
        _dones = torch.tensor(np.array(_dataset['dones']), dtype=torch.float, device=self.device)
        self.replay_buffer.add(_obs, _actions, _rewards, _next_obs, _dones)
        print('hdf5 loaded from', dataset_path, 'now idx', self.replay_buffer.idx)
        return _obs, _actions, _rewards, _next_obs, _dones

    def regression(self, train_dataset, batch_size=256, total_epoch_num=570):
        from torch.utils.data import Dataset, DataLoader, random_split
        
        self.init_tensors()
        self.algo_observer.after_init(self)
        self.last_mean_rewards = -100500
        total_time = 0
        # rep_count = 0
        self.frame = 0
        # self.obs = self.env_reset()
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)

        print('\033[1;33mStart training\033[0m')  # add hint

        while self.epoch_num < total_epoch_num:
            
            self.epoch_num += 1
            frame = self.epoch_num
            print('\033[1;32m---------------- Epoch {} ----------------\033[0m'.format(self.epoch_num))

            # train epoch
            actor_losses = []
            entropies = []
            alphas = []
            alpha_losses = []
            critic1_losses = []
            critic2_losses = []
            # add cql params
            min_qf1_losses = []
            min_qf2_losses = []
            std_q1s = []
            std_q2s = []
            alpha_prime_losses = []

            self.set_train()
            for s in train_loader:
                obs, reward, next_obs, done, action = s
                not_done = 1.0 - done.float()
                
                # update
                critic_loss, critic1_loss, critic2_loss, min_qf1_loss, min_qf2_loss, std_q1, std_q2, alpha_prime_loss \
                    = self.update_critic(obs, action, reward, next_obs, not_done, 0)

                actor_loss, entropy, alpha, alpha_loss = self.update_actor_and_alpha(obs, 0)

                actor_loss_info = actor_loss, entropy, alpha, alpha_loss
                self.soft_update_params(self.model.sac_network.critic, self.model.sac_network.critic_target,
                                        self.critic_tau)

                self.extract_actor_stats(actor_losses, entropies, alphas, alpha_losses, actor_loss_info)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)
                min_qf1_losses.append(min_qf1_loss)
                min_qf2_losses.append(min_qf2_loss)
                std_q1s.append(std_q1)
                std_q2s.append(std_q2)
                alpha_prime_losses.append(alpha_prime_loss)

            self.set_eval()
            eval_loss = []
            for s in train_loader:
                obs, reward, next_obs, done, action = s
                with torch.no_grad():
                    pred_action = self.act(obs.float(), self.env_info["action_space"].shape, sample=True)
                    loss = torch.norm(pred_action-action)
                    eval_loss.append(loss)
            mean_valid_loss = sum(eval_loss)/len(eval_loss)
            print(f'Epoch [{frame}/{total_epoch_num}]: Train loss: {actor_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')


            self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(actor_losses).item(), frame)
            self.writer.add_scalar('losses/c1_loss', torch_ext.mean_list(critic1_losses).item(), frame)
            self.writer.add_scalar('losses/c2_loss', torch_ext.mean_list(critic2_losses).item(), frame)
            # std Q value add
            self.writer.add_scalar('losses/std_c1_loss', torch_ext.mean_list(std_q1s).item(), frame)
            self.writer.add_scalar('losses/std_c2_loss', torch_ext.mean_list(std_q2s).item(), frame)
            self.writer.add_scalar('losses/min_c1_loss', torch_ext.mean_list(min_qf1_losses).item(), frame)
            self.writer.add_scalar('losses/min_c2_loss', torch_ext.mean_list(min_qf2_losses).item(), frame)
            if self.with_lagrange:
                self.writer.add_scalar('losses/alpha_prime_loss', torch_ext.mean_list(alpha_prime_losses).item(),
                                        frame)
            # end cql
            self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), frame)
            if alpha_losses[0] is not None:
                self.writer.add_scalar('losses/alpha_loss', torch_ext.mean_list(alpha_losses).item(), frame)
            self.writer.add_scalar('info/alpha', torch_ext.mean_list(alphas).item(), frame)

            self.writer.add_scalar('info/epochs', self.epoch_num, frame)
            self.algo_observer.after_print_stats(frame, self.epoch_num, total_time)

            # <editor-fold desc="Checkpoint">
            if mean_valid_loss < 1e-2:
                print('vaild loss: ', mean_valid_loss)
                self.save(
                    os.path.join(self.checkpoint_dir, 'reg_ep_' + str(self.epoch_num)))

            if self.epoch_num >= total_epoch_num:
                self.save(os.path.join(self.checkpoint_dir,
                                        'reg_last' + str(self.epoch_num)))
                print('MAX EPOCHS NUM!')

            # </editor-fold>
