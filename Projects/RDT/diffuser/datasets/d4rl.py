import os
import collections
import numpy as np
import gym
import pdb
from pathlib import Path

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

KEYS = ["observations","actions", "rewards", "terminals"]

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

# with suppress_output():
#     ## d4rl prints out a variety of warnings
#     import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#
def load_metaworld_episode(fn):
    # print(fn)
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in KEYS}
        return episode

def load_metaworld_dataset(replay_dir, task_list, isdebug=False, data_num=1000):
    data_ = collections.defaultdict(list)
    #save = [
    # print(task_list)
    if not isdebug:
        leng = data_num
    else:
        leng = 10
    for task_id in range(len(task_list)):       # loop
        _replay_dir = os.path.join(replay_dir, task_list[task_id])
        _replay_dir = Path(_replay_dir) #/ Path(os.listdir(_replay_dir)[0]) / 'dataset'
        # eps_fns = sorted(_replay_dir.glob('*.npz'), key=lambda x:int(str(x)))
        eps_fns = sorted(_replay_dir.glob('*.npz'), key=lambda x: int(x.stem.split('.')[0]))
        for eps_fn in [eps_fns[i] for i in range(leng)]:    #eps_fns[-20:]:#x:#eps_fns[-20:]:
            episode = load_metaworld_episode(eps_fn)
            for i in range(episode["terminals"].shape[0]):
                done_bool = bool(episode['terminals'][i])
                for k in episode:
                    data_[k].append(episode[k][i])
                if done_bool:
                    episode_data = {}
                    for k in data_:
                        episode_data[k] = np.array(data_[k])
                    # episode_data['task']= task_list[task_id]
                    yield episode_data
                    data_ = collections.defaultdict(list)



#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode