from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets
from rl_games.algos_torch import ppg_aux
from rl_games.common.ewma_model import EwmaModel

from torch import optim
import torch 
from torch import nn
import numpy as np
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
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        if self.ewma_ppo:
            self.ewma_model = EwmaModel(self.model, ewma_decay=0.889)
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_len' : self.seq_len,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std
        if 'phasic_policy_gradients' in self.config:
            self.has_phasic_policy_gradients = True
            self.ppg_aux_loss = ppg_aux.PPGAux(self, self.config['phasic_policy_gradients'])
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
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
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

            if self.ewma_ppo:
                ewma_dict = self.ewma_model(batch_dict)
                proxy_neglogp = ewma_dict['prev_neglogp']
                a_loss = common_losses.decoupled_actor_loss(old_action_log_probs_batch, action_log_probs, proxy_neglogp, advantage, curr_e_clip)
                old_action_log_probs_batch = proxy_neglogp # to get right statistic later
                old_mu_batch = ewma_dict['mus']
                old_sigma_batch = ewma_dict['sigmas']
            else:
                a_loss = common_losses.actor_loss(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            if self.add_cql:
                if self.cql_version == 0:
                    # CQL-H
                    cql_loss = torch.logsumexp(values, dim=1).mean() - values.mean()
                    c_loss = c_loss + cql_loss
                elif self.cql_version == 1:
                    # CQL-rho
                    cql_loss = F.normalize(torch.exp(values)) * values - values
                    c_loss = c_loss + cql_loss


            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        if self.ewma_ppo:
            self.ewma_model.update()                    

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu*mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def regression(self, train_dataset, batch_size=256, total_epoch_num=200):
        from torch.utils.data import Dataset, DataLoader, random_split
        import time
        import os
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)

        while self.epoch_num < total_epoch_num:
            epoch_num = self.update_epoch()
            print('\033[1;32m---------------- Epoch {} ----------------\033[0m'.format(epoch_num))
            # step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            self.vec_env.set_train_info(self.frame, self)
            if self.ewma_ppo:
                self.ewma_model.reset()
            self.set_eval()
            play_time_start = time.time()
            # with torch.no_grad():
            #     if self.is_rnn:
            #         batch_dict = self.play_steps_rnn()
            #     else:
            #         batch_dict = self.play_steps()

            update_list = self.update_list

            step_time = 0.0

            for n, s in enumerate(train_loader):
                _obs, _reward, _next_obs, _done, _action = s

                for idx in range(len(_obs)):
                    i = idx + batch_size * n
                    if i >= self.horizon_length:
                        break
                    obs = _obs[idx].view(-1, 74)
                    reward = _reward[idx].view(-1, 1)
                    next_obs = _next_obs[idx].view(-1, 74)
                    done = _done[idx].view(-1, 1)
                    action = _action[idx].view(-1, 18)

                    self.obs = obs
                    # get action value
                    processed_obs = self._preproc_obs(next_obs)
                    self.model.eval()
                    input_dict = {
                        'is_train': False,
                        'prev_actions': action,
                        'obs': processed_obs,
                        'rnn_states': self.rnn_states
                    }
                    with torch.no_grad():
                        res_dict = self.model(input_dict)
                    # return res_dict

                    self.experience_buffer.update_data('obses', i, obs)
                    self.experience_buffer.update_data('dones', i, done)
                    for k in update_list:
                        self.experience_buffer.update_data(k, i, res_dict[k])


                    shaped_rewards = self.rewards_shaper(reward)

                    self.experience_buffer.update_data('rewards', i, shaped_rewards)

                    self.current_lengths += 1
                    # all_done_indices = done.nonzero(as_tuple=False)
                    # env_done_indices = done.view(self.num_actors, self.num_agents).all(dim=1).nonzero(as_tuple=False)

                    not_dones = 1.0 - done.float()

                    self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
                    self.current_lengths = self.current_lengths * not_dones

            # last_values = self.get_values(self.obs)
            with torch.no_grad():
                self.model.eval()
                processed_obs = self._preproc_obs(obs)
                input_dict = {
                    'is_train': False,
                    'prev_actions': None,
                    'obs': processed_obs,
                    'rnn_states': self.rnn_states
                }
                result = self.model(input_dict)
                last_values = result['values']

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

            # return batch_dict


            # play steps end
            play_time_end = time.time()
            update_time_start = time.time()
            # rnn_masks = batch_dict.get('rnn_masks', None)

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

            # for mini_ep in range(0, self.mini_epochs_num):
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

                if self.schedule_type == 'legacy':
                    if self.multi_gpu:
                        kl = self.hvd.average_value(kl, 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef,
                                                                            self.epoch_num, 0, kl.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)

            kls.append(av_kls)
            # self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval()  # don't need to update statstics more than one miniepoch

            update_time_end = time.time()
            play_time = play_time_end - play_time_start
            update_time = update_time_end - update_time_start
            total_time = update_time_end - play_time_start

            # return batch_dict[
            #         'step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul


            # epoch end
            total_time += total_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False
            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                # do we need scaled_time?
                curr_frames = self.curr_frames
                self.frame += curr_frames
                # if self.print_stats:
                #     fps_step = curr_frames / step_time

                self.diagnostics.send_info(self.writer)
                self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(a_losses).item(), frame)
                self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(c_losses).item(), frame)

                self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), frame)
                self.writer.add_scalar('info/last_lr', last_lr * lr_mul, frame)
                self.writer.add_scalar('info/lr_mul', lr_mul, frame)
                self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
                self.writer.add_scalar('info/kl', torch_ext.mean_list(kls).item(), frame)
                self.writer.add_scalar('info/epochs', epoch_num, frame)
                self.algo_observer.after_print_stats(frame, epoch_num, total_time)

                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)
            mean_a_loss = sum(a_losses)/len(a_losses)
            print(f'Epoch [{epoch_num}/{total_epoch_num}]: Train loss: {mean_a_loss:.4f}')

            # if mean_a_loss < 1e-2:
            #     print('mean_a_loss : ', mean_a_loss)
            #     self.save(
            #         os.path.join(self.nn_dir, 'reg_ep_' + str(self.epoch_num)))
            #     should_exit = True

            if epoch_num >= total_epoch_num:
                self.save(os.path.join(self.nn_dir,
                                        'reglast_' + self.config['name'] + 'ep' + str(epoch_num)))
                print('MAX EPOCHS NUM!')
                should_exit = True

                update_time = 0

            if should_exit:
                return self.last_mean_rewards, epoch_num
    
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