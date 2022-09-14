from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets
from rl_games.algos_torch import network_builder
from rl_games.algos_torch import model_builder
from itertools import chain
from torch import optim
import torch
from torch import nn
import numpy as np
import time
import gym
from torch.nn import functional as F


def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class A2CAgent(a2c_common.ContinuousA2CBase):
    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        build_config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }

        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        print("model:", self.model)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound')  # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08,
                                    weight_decay=self.weight_decay)

        if self.has_central_value:
            cv_config = {
                'state_shape': self.state_shape,
                'value_size': self.value_size,
                'ppo_device': self.ppo_device,
                'num_agents': self.num_agents,
                'horizon_length': self.horizon_length,
                'num_actors': self.num_actors,
                'num_actions': self.actions_num,
                'seq_len': self.seq_len,
                'normalize_value': self.normalize_value,
                'network': self.central_value_config['network'],
                'config': self.central_value_config,
                'writter': self.writer,
                'max_epochs': self.max_epochs,
                'multi_gpu': self.multi_gpu,
                'hvd': self.hvd if self.multi_gpu else None
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn,
                                           self.ppo_device, self.seq_len)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = (self.has_central_value and self.use_experimental_cv) \
                              or (not self.has_phasic_policy_gradients and not self.has_central_value)
        self.algo_observer.after_init(self)

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo,
                                          curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch,
                                                   self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        # TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        self.diagnostics.mini_batch(self,
                                    {
                                        'values': value_preds_batch,
                                        'returns': return_batch,
                                        'new_neglogp': action_log_probs,
                                        'old_neglogp': old_action_log_probs_batch,
                                        'masks': rnn_masks
                                    }, curr_e_clip, 0)

        self.train_result = (a_loss, c_loss, entropy, \
                             kl_dist, self.last_lr, lr_mul, \
                             mu.detach(), sigma.detach(), b_loss)

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu * mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def load_hdf5(self, dataset_path):
        import h5py
        _dataset = h5py.File(dataset_path, 'r')
        _obs = torch.tensor(np.array(_dataset['observations']), dtype=torch.float, device=self.device)
        _actions = torch.tensor(np.array(_dataset['actions']), dtype=torch.float, device=self.device)
        _rewards = torch.tensor(np.array(_dataset['rewards']), dtype=torch.float, device=self.device)
        _next_obs = torch.tensor(np.array(_dataset['next_observations']), dtype=torch.float, device=self.device)
        _dones = torch.tensor(np.array(_dataset['dones']), dtype=torch.float, device=self.device)
        # self.replay_buffer.add(_obs, _actions, _rewards, _next_obs, _dones)
        # print('hdf5 loaded from', dataset_path, 'now idx', self.replay_buffer.idx)
        return _obs, _actions, _rewards, _next_obs, _dones


class A2CMultiAgent(a2c_common.ContinuousMultiA2CBase):
    def __init__(self, base_name, params):
        a2c_common.ContinuousMultiA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        self.load_SACnetworks(params)
        build_config = {
            'actions_num': int(self.actions_num / 2),
            'input_shape': obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.num_random = 1
        self.min_q_weight = self.config.get('min_q_weight')
        self.model_left = self.network.build(build_config)
        self.model_right = self.network.build(build_config)
        self.model_left.to(self.ppo_device)
        self.model_right.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model_left(self.model_left)
        self.init_rnn_from_model_right(self.model_right)
        self.last_lr_left = float(self.last_lr_left)
        self.last_lr_right = float(self.last_lr_right)

        self.alpha_lr = self.config.get('alpha_lr_rate')
        self.full_stage = self.config.get('full_stage', False)
        self.FrankaLimit = self.config.get('franka_limit')
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound')  # 'regularisation' or 'bound'
        self.optimizer_left = optim.Adam(self.model_left.parameters(), float(self.last_lr_left), eps=1e-08,
                                          weight_decay=self.weight_decay)
        self.optimizer_right = optim.Adam(self.model_right.parameters(), float(self.last_lr_right), eps=1e-08,
                                          weight_decay=self.weight_decay)
        self.Bimanual_regularization = self.config['Offline_PPO']['with_Bimanual_regularization']
        self.with_lagrange = self.config['Offline_PPO']['with_lagrange']
        self.offlinePPO = self.config['Offline_PPO']['offline_ppo']
        self.Double_Q = self.config['Offline_PPO']['DoubleQ']
        self.Double_Q = self.config['Offline_PPO']['DoubleQ']
        self.SAC_critic_lr = self.config['Offline_PPO']['critic_lr']
        self.SAC_actor_lr = self.config['Offline_PPO']['actor_lr']
        self.DoubleQ_rewardscale = self.config['Offline_PPO']['DoubleQ_rewardscale']
        if self.Double_Q:
            sacnet_config = {
                'obs_dim': 37,
                'action_dim': 9,
                'actions_num': int(self.actions_num / 2),
                'input_shape': obs_shape,
                'normalize_input': self.normalize_input,
            }
            self.sacmodel_left = self.sac_network.build(sacnet_config)
            self.sacmodel_right = self.sac_network.build(sacnet_config)
            self.sacmodel_left.to(self.ppo_device)
            self.sacmodel_right.to(self.ppo_device)
            self.sac_Closs = nn.MSELoss()
            self.sac_CriticOptimizer_left = torch.optim.Adam(self.sacmodel_left.sac_network.critic.parameters(),
                                                             lr=self.SAC_critic_lr,
                                                             betas=self.config.get("critic_betas", [0.9, 0.999]))
            self.sac_CriticOptimizer_right = torch.optim.Adam(self.sacmodel_right.sac_network.critic.parameters(),
                                                              lr=self.SAC_critic_lr,
                                                              betas=self.config.get("critic_betas", [0.9, 0.999]))
            self.sac_Actoroptimizer_left = torch.optim.Adam(self.sacmodel_left.sac_network.actor.parameters(),
                                                            lr=self.SAC_actor_lr,
                                                            betas=self.config.get("critic_betas", [0.9, 0.999]))
            self.sac_Actoroptimizer_right = torch.optim.Adam(self.sacmodel_right.sac_network.actor.parameters(),
                                                            lr=self.SAC_actor_lr,
                                                            betas=self.config.get("critic_betas", [0.9, 0.999]))
        if self.has_central_value:
            cv_config = {
                'state_shape': self.state_shape,
                'value_size': self.value_size,
                'ppo_device': self.ppo_device,
                'num_agents': self.num_agents,
                'horizon_length': self.horizon_length,
                'num_actors': self.num_actors,
                'num_actions': self.actions_num,
                'seq_len': self.seq_len,
                'normalize_value': self.normalize_value,
                'network': self.central_value_config['network'],
                'config': self.central_value_config,
                'writter': self.writer,
                'max_epochs': self.max_epochs,
                'multi_gpu': self.multi_gpu,
                'hvd': self.hvd if self.multi_gpu else None
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        # self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        self.dataset_left = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete,
                                                     self.is_rnn_left, self.ppo_device, self.seq_len)
        self.dataset_offline_left = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete,
                                                             self.is_rnn_left, self.ppo_device, self.seq_len)
        self.dataset_right = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete,
                                                       self.is_rnn_right, self.ppo_device, self.seq_len)
        self.dataset_offline_right = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete,
                                                               self.is_rnn_right, self.ppo_device, self.seq_len)
        if self.normalize_value:
            self.value_mean_std_left = self.central_value_net.model.value_mean_std if self.has_central_value else self.model_left.value_mean_std
            self.value_mean_std_right = self.central_value_net.model.value_mean_std if self.has_central_value else self.model_right.value_mean_std

        self.has_value_loss = (self.has_central_value and self.use_experimental_cv) \
                              or (not self.has_phasic_policy_gradients and not self.has_central_value)  # True
        self.algo_observer_left.after_init(self)
        self.algo_observer_right.after_init(self)

        if self.with_lagrange:
            self.target_action_gap_left = self.config.get('lagrange_thresh_left')
            self.target_action_gap_right = self.config.get('lagrange_thresh_right')
            self.log_alpha_prime_left = torch.zeros(1, requires_grad=True, device=self.ppo_device)
            self.log_alpha_prime_right = torch.zeros(1, requires_grad=True, device=self.ppo_device)
            self.alpha_prime_optimizer_left = torch.optim.Adam([self.log_alpha_prime_left],
                                                               lr=self.alpha_lr
                                                               )
            self.alpha_prime_optimizer_right = torch.optim.Adam([self.log_alpha_prime_right],
                                                                lr=self.alpha_lr
                                                                )

    def bimanual_regularization(self, current_actions, offline_actions, offline_obs):
        # update obs, action, reward, next_obs, done from replay_buffer(one step)

        offline_actions_data = offline_actions.expand(current_actions.size(0), offline_actions.size(0), 9)
        current_actions = torch.unsqueeze(current_actions, 1)
        dis = torch.norm((offline_actions_data - current_actions), dim=2)
        min = torch.argmin(dis, axis=1)
        min_numpy = np
        # print(min.size())
        # print(min, dis[min])
        current_action = offline_actions[min]
        current_obs = offline_obs[min]

        return current_action, current_obs

    def load_SACnetworks(self, params):
        sac_params = params
        sac_params['algo']['name'] = 'sac'
        sac_params['network']['name'] = 'soft_actor_critic'
        sac_params['network']['separate'] = 'True'
        sac_params['model']['name'] = 'soft_actor_critic'
        builder = model_builder.ModelBuilder()
        self.sac_network = builder.load(sac_params)

    def create_random_action(self, values_offline, actions_batch):
        random_actions_tensor = torch.FloatTensor(values_offline.size(0) *
                                                  self.num_random, actions_batch.shape[-1]).to(
            self.ppo_device)

        for i in range(random_actions_tensor.size()[1]):
            random_actions_tensor[:, i].uniform_(self.FrankaLimit[1][i], self.FrankaLimit[0][i]).size()

        return random_actions_tensor

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def save_multi(self, fn):
        state_left, state_right = self.get_full_state_weights()
        save_model = {
            'model_left': state_left,
            'model_right': state_right
        }
        torch_ext.save_checkpoint(fn, save_model)

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        # change, q1,q2=self.model.critic(obs,action)
        preds_q1, preds_q2 = network(obs_temp, actions)
        preds_q1 = preds_q1.view(obs.shape[0], num_repeat, 1)
        return preds_q1, preds_q2

    def calc_gradients_left(self, input_dict, data_actions_left, data_obs_left, data_next_obs_left,
                            data_next_actions_left, data_rewards_left, data_dones_left):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        obs_batch_offline = self._preproc_obs(data_obs_left)
        next_obs_batch_offline = self._preproc_obs(data_next_obs_left)
        lr_mul = 1.0
        curr_e_clip = self.e_clip

        if self.Bimanual_regularization:
            # todo
            self.bimanual_loss = nn.MSELoss()
            actions_batch, obs_batch = self.bimanual_regularization(
                actions_batch, data_actions_left, obs_batch_offline)

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }


        rnn_masks = None
        if self.is_rnn_left:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model_left(batch_dict)
            # res_dict_offline = self.sacmodel_left.actor(data_obs_left)
            # current_actions = res_dict_offline.rsample()
            if self.Double_Q:
                next_res_dict_offline = self.sacmodel_left.actor(data_next_obs_left)
                next_actions = next_res_dict_offline.rsample()
            action_log_probs = res_dict['prev_neglogp']
            Qvalues = res_dict['values']

            # Qvalues_offline_next = next_res_dict_offline['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            ######################################################
            if self.offlinePPO:
                # add CQL and DoubleQ function
                # random_actions_tensor = self.create_random_action(data_actions_left, actions_batch)
                q1_pred, q2_pred = self.sacmodel_left.critic(obs_batch_offline, data_actions_left)
                # q1_random, q2_random = self.sacmodel_left.critic(obs_batch_offline, random_actions_tensor)
                # q1_current, q2_current = self.sacmodel_left.critic(obs_batch_offline, current_actions)
                # q1_next, q2_next = self.sacmodel_left.critic(obs_batch_offline, next_actions)
                q1_target, q2_target = self.sacmodel_left.critic_target(next_obs_batch_offline, next_actions)
                target_q_values = torch.min(q1_target, q2_target)
                q_target = self.DoubleQ_rewardscale * data_rewards_left + (
                        1 - data_dones_left) * self.DoubleQ_rewardscale * target_q_values
                q_target.detach()
                q1_loss = self.sac_Closs(q1_pred, q_target)
                q2_loss = self.sac_Closs(q2_pred, q_target)
                cat_q1 = torch.cat([q1_pred], 1)
                cat_q2 = torch.cat([q2_pred], 1)
                # logsumexp= Log(Sum(Exp()))
                min_qf1_loss = torch.logsumexp(cat_q1 / 1.0, dim=1, ).mean() * self.min_q_weight
                min_qf2_loss = torch.logsumexp(cat_q2 / 1.0, dim=1, ).mean() * self.min_q_weight
                # min_qf1_loss = min_qf1_loss - q1_pred.mean()
                # min_qf2_loss = min_qf2_loss - q2_pred.mean()

                if self.with_lagrange:
                    alpha_prime = torch.clamp(self.log_alpha_prime_left.exp(), min=0.0, max=10.0)
                    min_qf1_loss_alpha = alpha_prime * (min_qf1_loss - self.target_action_gap_left)
                    min_qf2_loss_alpha = alpha_prime * (min_qf2_loss - self.target_action_gap_left)
                    alpha_prime_loss = (-min_qf1_loss_alpha - min_qf2_loss_alpha) * 0.5

                """Subtract the log likelihood of data"""
                q1_loss = q1_loss + min_qf1_loss_alpha.detach()
                q2_loss = q2_loss + min_qf2_loss_alpha.detach()
                q_loss = q1_loss + q2_loss
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo,
                                          curr_e_clip)
            if self.has_value_loss:
                # c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch,
                #                                    self.clip_value)
                c_loss = common_losses.critic_loss(value_preds_batch, Qvalues, curr_e_clip, return_batch,
                                                   self.clip_value)
                if self.offlinePPO:
                    c_loss = c_loss + (min_qf1_loss_alpha.detach() + min_qf2_loss_alpha.detach()) * 0.5
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef_left + b_loss * self.bounds_loss_coef

            if self.multi_gpu:
                self.optimizer_left.zero_grad()
            else:
                for param in self.model_left.parameters():
                    param.grad = None

        if self.with_lagrange:
            self.alpha_prime_optimizer_left.zero_grad()
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer_left.step()
        if self.Double_Q:
            self.sac_CriticOptimizer_left.zero_grad(set_to_none=True)
            q_loss.backward(retain_graph=True)
            self.sac_CriticOptimizer_left.step()

        # TODO: Refactor this ugliest code of they year
        self.optimizer_left.zero_grad(set_to_none=True)
        self.scaler_left.scale(loss).backward()
        self.trancate_gradients_and_step_left()
        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        self.diagnostics.mini_batch(self,
                                    {
                                        'values': value_preds_batch,
                                        'returns': return_batch,
                                        'new_neglogp': action_log_probs,
                                        'old_neglogp': old_action_log_probs_batch,
                                        'masks': rnn_masks
                                    }, curr_e_clip, 0)
        if self.offlinePPO:
            self.train_result_left = (a_loss, c_loss, entropy, \
                                      kl_dist, self.last_lr_left, lr_mul, \
                                      mu.detach(), sigma.detach(), b_loss.detach(), min_qf1_loss_alpha.detach(),
                                      alpha_prime.mean().detach(), min_qf1_loss.detach())
        else:
            self.train_result_left = (a_loss, c_loss, entropy, \
                                      kl_dist, self.last_lr_left, lr_mul, \
                                      mu.detach(), sigma.detach(), b_loss)

    def calc_gradients_right(self, input_dict, data_actions_right, data_obs_right, data_next_obs_right,
                             data_next_actions_right, data_rewards_right, data_dones_right):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        obs_batch_offline = self._preproc_obs(data_obs_right)
        next_obs_batch_offline = self._preproc_obs(data_next_obs_right)
        lr_mul = 1.0
        curr_e_clip = self.e_clip

        if self.Bimanual_regularization:
            # todo
            self.bimanual_loss = nn.MSELoss()
            actions_batch, obs_batch = self.bimanual_regularization(
                actions_batch, data_actions_right, obs_batch_offline)

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

        rnn_masks = None
        if self.is_rnn_right:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model_right(batch_dict)
            # res_dict_offline = self.sacmodel_right.actor(data_obs_right)
            # current_actions = res_dict_offline.rsample()
            if self.Double_Q:
                next_res_dict_offline = self.sacmodel_right.actor(data_next_obs_right)
                next_actions = next_res_dict_offline.rsample()
            action_log_probs = res_dict['prev_neglogp']
            Qvalues = res_dict['values']

            # Qvalues_offline_next = next_res_dict_offline['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']

            sigma = res_dict['sigmas']
            ######################################################
            if self.offlinePPO:
                # add CQL and DoubleQ function
                # random_actions_tensor = self.create_random_action(data_actions_right, actions_batch)
                q1_pred, q2_pred = self.sacmodel_right.critic(obs_batch_offline, data_actions_right)
                # q1_random, q2_random = self.sacmodel_right.critic(obs_batch_offline, random_actions_tensor)
                # q1_next, q2_next = self.sacmodel_right.critic(obs_batch_offline, next_actions)
                # q1_current, q2_current = self.sacmodel_right.critic(obs_batch_offline, current_actions)
                q1_target, q2_target = self.sacmodel_right.critic_target(next_obs_batch_offline, next_actions)
                target_q_values = torch.min(q1_target, q2_target)
                q_target = self.DoubleQ_rewardscale * data_rewards_right + (
                        1 - data_dones_right) * self.DoubleQ_rewardscale * target_q_values
                q_target.detach()
                q1_loss = self.sac_Closs(q1_pred, q_target)
                q2_loss = self.sac_Closs(q2_pred, q_target)
                cat_q1 = torch.cat([q1_pred], 1)
                cat_q2 = torch.cat([q2_pred], 1)
                # logsumexp= Log(Sum(Exp()))
                min_qf1_loss = torch.logsumexp(cat_q1 / 1.0, dim=1, ).mean() * self.min_q_weight
                min_qf2_loss = torch.logsumexp(cat_q2 / 1.0, dim=1, ).mean() * self.min_q_weight
                min_qf1_loss = min_qf1_loss - q1_pred.mean()
                min_qf2_loss = min_qf2_loss - q2_pred.mean()

                if self.with_lagrange:
                    alpha_prime = torch.clamp(self.log_alpha_prime_right.exp(), min=0.0, max=10.0)
                    min_qf1_loss_alpha = alpha_prime * (min_qf1_loss - self.target_action_gap_right)
                    min_qf2_loss_alpha = alpha_prime * (min_qf2_loss - self.target_action_gap_right)
                    alpha_prime_loss = (-min_qf1_loss_alpha - min_qf2_loss_alpha) * 0.5

                """Subtract the log likelihood of data"""
                q1_loss = q1_loss + min_qf1_loss_alpha.detach()
                q2_loss = q2_loss + min_qf2_loss_alpha.detach()
                q_loss = q1_loss + q2_loss
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo,
                                          curr_e_clip)
            if self.has_value_loss:
                # c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch,
                #                                    self.clip_value)
                c_loss = common_losses.critic_loss(value_preds_batch, Qvalues, curr_e_clip, return_batch,
                                                   self.clip_value)
                if self.offlinePPO:
                    c_loss = c_loss + (min_qf1_loss_alpha.detach() + min_qf2_loss_alpha.detach()) * 0.5
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef_right + b_loss * self.bounds_loss_coef

            if self.multi_gpu:
                self.optimizer_right.zero_grad()
            else:
                for param in self.model_right.parameters():
                    param.grad = None

        if self.with_lagrange:
            self.alpha_prime_optimizer_right.zero_grad(set_to_none=True)
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer_right.step()
        if self.Double_Q:
            self.sac_CriticOptimizer_right.zero_grad(set_to_none=True)
            q_loss.backward(retain_graph=True)
            self.sac_CriticOptimizer_right.step()

        # TODO: Refactor this ugliest code of they year
        self.optimizer_right.zero_grad(set_to_none=True)
        self.scaler_right.scale(loss).backward()
        self.trancate_gradients_and_step_right()
        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        self.diagnostics.mini_batch(self,
                                    {
                                        'values': value_preds_batch,
                                        'returns': return_batch,
                                        'new_neglogp': action_log_probs,
                                        'old_neglogp': old_action_log_probs_batch,
                                        'masks': rnn_masks
                                    }, curr_e_clip, 0)
        if self.offlinePPO:
            self.train_result_right = (a_loss, c_loss, entropy, \
                                       kl_dist, self.last_lr_right, lr_mul, \
                                       mu.detach(), sigma.detach(), b_loss, min_qf1_loss_alpha,
                                       alpha_prime.mean(), min_qf1_loss)
        else:
            self.train_result_right = (a_loss, c_loss, entropy, \
                                       kl_dist, self.last_lr_right, lr_mul, \
                                       mu.detach(), sigma.detach(), b_loss)

    def train_actor_critic_multi(self, input_dict_left, input_dict_right,
                                 data_actions_left, data_obs_left, data_next_obs_left,
                                 data_next_actions_left, data_rewards_left, data_dones_left,
                                 data_actions_right, data_obs_right, data_next_obs_right,
                                 data_next_actions_right, data_rewards_right, data_dones_right):
        self.calc_gradients_left(input_dict_left, data_actions_left, data_obs_left,
                                 data_next_obs_left, data_next_actions_left, data_rewards_left, data_dones_left)
        self.calc_gradients_right(input_dict_right, data_actions_right, data_obs_right,
                                  data_next_obs_right, data_next_actions_right, data_rewards_right, data_dones_right)
        self.train_result = self.train_result_left + self.train_result_right
        return self.train_result

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu * mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def load_hdf5(self, dataset_path):
        import h5py
        _dataset = h5py.File(dataset_path, 'r')
        _obs = torch.tensor(np.array(_dataset['observations']), dtype=torch.float, device=self.device)
        _actions = torch.tensor(np.array(_dataset['actions']), dtype=torch.float, device=self.device)
        _rewards = torch.tensor(np.array(_dataset['rewards']), dtype=torch.float, device=self.device)
        _next_obs = torch.tensor(np.array(_dataset['next_observations']), dtype=torch.float, device=self.device)
        _dones = torch.tensor(np.array(_dataset['dones']), dtype=torch.float, device=self.device)
        # self.replay_buffer.add(_obs, _actions, _rewards, _next_obs, _dones)
        # print('hdf5 loaded from', dataset_path, 'now idx', self.replay_buffer.idx)
        return _obs, _actions, _rewards, _next_obs, _dones
