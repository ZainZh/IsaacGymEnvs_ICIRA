from rl_games.common.player import BasePlayer
from rl_games.common.player import BaseMultiPlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.tr_helpers import unsqueeze_obs
import gym
import torch 
from torch import nn
import numpy as np


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action


class PpoPlayerContinuous(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        } 
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_action(self, obs, is_determenistic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def reset(self):
        self.init_rnn()

class PpoMultiPlayerContinuous(BaseMultiPlayer):
    def __init__(self, params):
        BaseMultiPlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        # self.actions_low = torch.cat((self.actions_low, self.actions_low), 0)
        # self.actions_high = torch.cat((self.actions_high, self.actions_high), 0)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.model_left = self.network.build(config)
        self.model_right = self.network.build(config)
        self.model_left.to(self.device)
        self.model_right.to(self.device)
        self.model_left.eval()
        self.model_right.eval()
        self.is_rnn_left = self.model_left.is_rnn()
        self.is_rnn_right = self.model_right.is_rnn()

    def action_split(self, actions):
        actions_left = actions[:, 0:9]
        actions_right = actions[:, 9:18]
        return actions_left, actions_right

    def action_combine(self, actions_left, actions_right):
        actions = torch.cat((actions_left, actions_right), 1)
        return actions

    def get_action_left(self, obs, is_determenistic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : obs,
            'rnn_states' : self.states_left
        }
        with torch.no_grad():
            res_dict_left = self.model_left(input_dict)

        mu_left = res_dict_left['mus']

        action_left = res_dict_left['actions']

        self.states_left = res_dict_left['rnn_states']

        if is_determenistic:
            current_action = mu_left
        else:
            current_action = action_left
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def get_action_right(self, obs, is_determenistic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : obs,
            'rnn_states' : self.states_right
        }
        with torch.no_grad():

            res_dict_right = self.model_right(input_dict)

        mu_right = res_dict_right['mus']

        action_right = res_dict_right['actions']
        self.states_right = res_dict_right['rnn_states']

        if is_determenistic:
            current_action =mu_right
        else:
            current_action =action_right
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        if 'a2c_network.critic_mlp.4.weight' in checkpoint['model_left']['model']:
            del checkpoint['model_left']['model']['a2c_network.critic_mlp.0.weight']
            del checkpoint['model_left']['model']['a2c_network.critic_mlp.2.weight']
            del checkpoint['model_left']['model']['a2c_network.critic_mlp.4.weight']
            del checkpoint['model_left']['model']['a2c_network.critic_mlp.0.bias']
            del checkpoint['model_left']['model']['a2c_network.critic_mlp.2.bias']
            del checkpoint['model_left']['model']['a2c_network.critic_mlp.4.bias']

            del checkpoint['model_right']['model']['a2c_network.critic_mlp.0.weight']
            del checkpoint['model_right']['model']['a2c_network.critic_mlp.2.weight']
            del checkpoint['model_right']['model']['a2c_network.critic_mlp.4.weight']
            del checkpoint['model_right']['model']['a2c_network.critic_mlp.0.bias']
            del checkpoint['model_right']['model']['a2c_network.critic_mlp.2.bias']
            del checkpoint['model_right']['model']['a2c_network.critic_mlp.4.bias']
        self.model_left.load_state_dict(checkpoint['model_left']['model'])
        self.model_right.load_state_dict(checkpoint['model_right']['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint['model_left']:
            self.model_left.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if self.normalize_input and 'running_mean_std' in checkpoint['model_right']:
            self.model_right.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def reset(self):
        self.init_rnn_multi()

class PpoPlayerDiscrete(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)

        self.network = self.config['network']
        if type(self.action_space) is gym.spaces.Discrete:
            self.actions_num = self.action_space.n
            self.is_multi_discrete = False
        if type(self.action_space) is gym.spaces.Tuple:
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True
        self.mask = [False]
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }

        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_masked_action(self, obs, action_masks, is_determenistic = True):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        action_masks = torch.Tensor(action_masks).to(self.device).bool()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'action_masks' : action_masks,
            'rnn_states' : self.states
        }
        self.model.eval()

        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict['logits']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if self.is_multi_discrete:
            if is_determenistic:
                action = [torch.argmax(logit.detach(), axis=-1).squeeze() for logit in logits]
                return torch.stack(action,dim=-1)
            else:    
                return action.squeeze().detach()
        else:
            if is_determenistic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:    
                return action.squeeze().detach()

    def get_action(self, obs, is_determenistic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict['logits']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if self.is_multi_discrete:
            if is_determenistic:
                action = [torch.argmax(logit.detach(), axis=1).squeeze() for logit in logits]
                return torch.stack(action,dim=-1)
            else:    
                return action.squeeze().detach()
        else:
            if is_determenistic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:    
                return action.squeeze().detach()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def reset(self):
        self.init_rnn()


class SACPlayer(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = self.obs_shape
        self.normalize_input = False
        config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': False,
            'normalize_input': self.normalize_input,
        }  
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.sac_network.actor.load_state_dict(checkpoint['actor'])
        self.model.sac_network.critic.load_state_dict(checkpoint['critic'])
        self.model.sac_network.critic_target.load_state_dict(checkpoint['critic_target'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def get_action(self, obs, is_determenistic=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        dist = self.model.actor(obs)
        actions = dist.sample() if is_determenistic else dist.mean
        actions = actions.clamp(*self.action_range).to(self.device)
        if self.has_batch_dimension == False:
            actions = torch.squeeze(actions.detach())
        return actions

    def reset(self):
        pass


class CQLPlayer(SACPlayer):
    def __init__(self,params):
        # self.state_shape = 42
        super().__init__(params)