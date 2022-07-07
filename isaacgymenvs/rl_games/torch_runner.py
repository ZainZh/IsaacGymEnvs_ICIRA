import time
import numpy as np
import random
import copy
import torch
import yaml
import os

from rl_games import envs
from rl_games.common import object_factory
from rl_games.common import env_configurations
from rl_games.common import experiment
from rl_games.common import tr_helpers

from rl_games.algos_torch import model_builder
from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent
import rl_games.networks
from rl_games.algos_torch import cql_agent
from torch.utils.data import Dataset, DataLoader, random_split

def _restore(agent, args):
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        agent.restore(args['checkpoint'])
    else:
        # no restore but model exists
        basepath = agent.nn_dir+agent.experiment_name+'.pth'
        if os.path.exists(basepath):
            raise Exception('pth exists!', basepath)

def _override_sigma(agent, args):
    if 'sigma' in args and args['sigma'] is not None:
        net = agent.model.sac_network   # a2c_network -> sac_network
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(args['sigma']))
            else:
                print('Print cannot set new sigma because fixed_sigma is False')

def _load_hdf5(agent, args):
    if args['dataset'] is not None and args['dataset']!='':
        return agent.load_hdf5(args['dataset'])


class Runner:
    def __init__(self, algo_observer=None):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_continuous_multi', lambda **kwargs : a2c_continuous.A2CMultiAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs))
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))
        #self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))
        self.algo_factory.register_builder('cql', lambda **kwargs : cql_agent.CQLAgent(**kwargs))   #add cql builder

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))
        #self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))
        self.player_factory.register_builder('cql', lambda **kwargs : players.CQLPlayer(**kwargs)) #add cql builder

        self.algo_observer = algo_observer if algo_observer else DefaultAlgoObserver()
        torch.backends.cudnn.benchmark = True
        ### it didnot help for lots for openai gym envs anyway :(
        #torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)
    def reset(self):
        pass

    def load_config(self, params):
        self.seed = params.get('seed', None)
        if self.seed is None:
            self.seed = int(time.time())
        
        if params["config"].get('multi_gpu', False):
            import horovod.torch as hvd

            hvd.init()
            self.seed += hvd.rank()
        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            
            # deal with environment specific seed if applicable
            if 'env_config' in params['config']:
                if not 'seed' in params['config']['env_config']:
                    params['config']['env_config']['seed'] = self.seed
                else:
                    if params["config"].get('multi_gpu', False):
                        params['config']['env_config']['seed'] += hvd.rank()

        config = params['config']
        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer     # here algo_observer
        self.params = params

    def load(self, yaml_conf):
        self.default_config = yaml_conf['params']
        self.load_config(params=copy.deepcopy(self.default_config))

    def run_train(self, args):
        print('\033[1;33mStarted to train\033[0m')
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        _restore(agent, args)
        if self.algo_name == 'cql' or self.algo_name == 'sac':
            _override_sigma(agent, args)
            _load_hdf5(agent, args)
        agent.train()

    def run_play(self, args):
        print('Started to play')
        player = self.create_player()
        _restore(player, args)
        if self.algo_name == 'cql':
            _override_sigma(player, args)
        player.run()

    def create_player(self):
        return self.player_factory.create(self.algo_name, params=self.params)

    def reset(self):
        pass

    def run(self, args):
        load_path = None

        if args['train']:
            self.run_train(args)

        elif args['play']:
            self.run_play(args)
        elif args['reg']:
            if args['dataset'] is not None and args['dataset']!='':
                self.train_regression(args)
            else:
                raise Exception('add dataset path!')
        else:
            self.run_train(args)

    def train_regression(self, args):
        class myhdf5dataset(Dataset):
            def __init__(self, obs, reward, next_obs, dones, action=None, device=args['device']):
                self.action = action
                self.obs = obs
                self.reward = reward
                self.next_obs = next_obs
                self.dones = dones

            def __getitem__(self, idx):
                if self.action is None:
                    return self.obs[idx], self.reward[idx], self.next_obs[idx], self.dones[idx]
                else:
                    return self.obs[idx], self.reward[idx], self.next_obs[idx], self.dones[idx], self.action[idx]

            def __len__(self):
                return len(self.obs)

        print('\033[1;33mTrain regression\033[0m')
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        if hasattr(agent.regression, '__call__'):
            _restore(agent, args)
            _obs, _actions, _rewards, _next_obs, _dones = _load_hdf5(agent, args)
            train_dataset = myhdf5dataset(_obs, _rewards, _next_obs, _dones, _actions)
            if self.algo_name == 'cql' or self.algo_name == 'sac':
                _override_sigma(agent, args)
                print("dataset learning start")
            agent.regression(train_dataset, batch_size=256)
        else:
            raise Exception('no this function')

        