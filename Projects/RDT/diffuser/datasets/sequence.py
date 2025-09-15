from collections import namedtuple
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
from .d4rl import load_metaworld_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
Batch = namedtuple('Batch', 'trajectories conditions')
ActBatch = namedtuple('ActBatch', 'actions conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


TASKLIST_10 = ['basketball-v2', 'button-press-v2', 'dial-turn-v2', 'drawer-close-v2', 'peg-insert-side-v2', 'pick-place-v2',
               'push-v2', 'reach-v2', 'window-open-v2', 'sweep-into-v2']

TASKLIST = ['basketball-v2', 'bin-picking-v2',  'button-press-topdown-v2',
 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2',
'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2',
'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2',
 'faucet-close-v2',  'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2',
 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2',
 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2',
 'plate-slide-back-side-v2',  'soccer-v2', 'push-wall-v2',  'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2',
'window-close-v2','assembly-v2','button-press-topdown-wall-v2','hammer-v2','peg-unplug-side-v2',
                               'reach-wall-v2', 'stick-push-v2', 'stick-pull-v2', 'box-close-v2']

TASKLIST_easy = ['button-press-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-wall-v2', 'coffee-button-v2',
                 'dial-turn-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2', 'drawer-close-v2', 'drawer-open-v2',
                 'faucet-close-v2', 'faucet-open-v2', 'handle-press-v2', 'handle-press-side-v2', 'handle-pull-v2', 'handle-pull-side-v2',
                 'lever-pull-v2', 'plate-slide-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'plate-slide-side-v2', 'reach-v2',
                 'reach-wall-v2', 'window-close-v2', 'window-open-v2', 'peg-unplug-side-v2']

TASKLIST_medium = ['basketball-v2', 'bin-picking-v2', 'box-close-v2', 'coffee-pull-v2', 'coffee-push-v2', 'hammer-v2','peg-insert-side-v2',
                   'push-wall-v2', 'soccer-v2', 'sweep-v2', 'sweep-into-v2']

TASKLIST_hard = ['assembly-v2', 'hand-insert-v2', 'pick-out-of-hole-v2', 'pick-place-v2', 'push-v2', 'push-back-v2']

TASKLIST_veryhard = ['shelf-place-v2', 'disassemble-v2', 'stick-pull-v2', 'stick-push-v2', 'pick-place-wall-v2']

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, env='hopper-medium-replay', horizon=64, pad_before=0, pad_after=0, To=2,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000, data_num=1000,
        max_n_episodes=80000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env_name = env
        self.datanum = data_num
        # self.env = env = load_environment(env)
        self.dataset_path = dataset_path
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.To = To
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        # itr = sequence_dataset(env, self.preprocess_fn)
        isdebug = self.dataset_path.endswith('_debug')
        ismt10 = (env == 'metaworld10')
        iseay = (env == 'metaworldeasy')
        ismed = (env == 'metaworldmedium')
        ishard = (env == 'metaworldhard')
        isvhard = (env == 'metaworldveryhard')
        if ismt10:
            tasklist = TASKLIST_10
        elif iseay:
            tasklist = TASKLIST_easy
        elif ismed:
            tasklist = TASKLIST_medium
        elif ishard:
            tasklist = TASKLIST_hard
        elif isvhard:
            tasklist = TASKLIST_veryhard
        else:
            tasklist = [env]        # single task
        itr = load_metaworld_dataset(self.dataset_path, tasklist, isdebug, data_num)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon, pad_before, pad_after)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon, pad_before=0, pad_after=0):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        pad_before = min(max(pad_before, 0), horizon-1)
        pad_after = min(max(pad_after, 0), horizon-1)
        indices = []
        for i, path_length in enumerate(path_lengths):
            min_start = -pad_before
            max_start = path_length - horizon + pad_after
            # max_start = min(path_length - 1, self.max_path_length - horizon)
            # if not self.use_padding:
            #     max_start = min(max_start, path_length - horizon)
            for idx in range(min_start, max_start+1):
                start = max(idx, 0)
                end = min(idx + horizon, path_length)
                start_offset = start - idx
                end_offset = (idx + horizon) - end
                sample_start_idx = 0 + start_offset                                                        
                sample_end_idx = horizon - end_offset
                indices.append((i, start, end, sample_start_idx, sample_end_idx))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[:self.To]}  

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end, sample_start_idx, sample_end_idx = self.indices[idx]

        obs_samples = self.fields.normed_observations[path_ind, start:end]
        act_samples = self.fields.normed_actions[path_ind, start:end]
        
        observations = obs_samples
        actions = act_samples
            
        if (sample_start_idx > 0) or (sample_end_idx < self.horizon):  
            observations = np.zeros(
                shape=(self.horizon,) + self.fields.observations.shape[-1:],
                dtype=self.fields.observations.dtype)                                       # (10,39)
            actions = np.zeros(
                    shape=(self.horizon,) + self.fields.actions.shape[-1:],
                    dtype=self.fields.actions.dtype)                                   # (10,4)                           
            if sample_start_idx > 0:                        # padding episode beginning
                observations[:sample_start_idx] = obs_samples[0]
                actions[:sample_start_idx] = act_samples[0]
            if sample_end_idx < self.horizon:       # padding episode end
                observations[sample_end_idx:] = obs_samples[-1]
                actions[sample_end_idx:] = act_samples[-1]
            observations[sample_start_idx:sample_end_idx] = obs_samples
            actions[sample_start_idx:sample_end_idx] = act_samples

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            # batch = Batch(trajectories, conditions)
            batch = ActBatch(actions, conditions)

        return batch