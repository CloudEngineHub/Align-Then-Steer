# import sys
#
# sys.path.append('E:\\vscode-insider\\Decision-diffuser\\code')
import os
import copy
import statistics

import numpy as np
import torch
import einops
import pdb
import diffuser
from copy import deepcopy
from typing import Optional, Dict
import pathlib
from diffuser.utils.arrays import batch_to_device, to_np, to_torch, to_device, apply_dict
from diffuser.utils.timer import Timer
# from .cloud import sync_logs
from ml_logger import logger
import wandb
# import wandb
import metaworld


def cycle(dl):
    while True:
        for data in dl:         # 在这里会调用sequence.__getitem__生成数据
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        To=2,
        Ta=8,
        #renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,       # 100
        log_dir=None,
        save_freq=1000,
        bucket=None,
        ckpt_name=None,
        train_device='cuda',
    ):
        super().__init__()

        wandb.init(project='MT50_100k',
                   name=f'{dataset.env_name}_pre_{dataset.datanum}',
                   dir='/home/your/wandb')
        wandb.define_metric("custom_step")

        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.save_freq = save_freq

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        # self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.bucket = bucket
        self.ckpt_name = ckpt_name

        self.reset_parameters()
        self.step = 0
        self.To = To
        self.Ta = Ta

        self.device = train_device

        # logger.print(self.dataset.fields)
        self.log_dir = log_dir

        loadpath = os.path.join(self.bucket, f'checkpoint/{self.ckpt_name}')
        if os.path.exists(loadpath):
            self.load()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    # one epoch
    def train(self, n_train_steps):
    # def train(self, n_train_steps, curr_epoch):
        # timer = Timer()
        self.model.train()
        # if curr_epoch * n_train_steps <= self.step < (curr_epoch + 1) * n_train_steps:
        #     remain_steps = (curr_epoch + 1) * n_train_steps - self.step
        for step in range(n_train_steps):
            if self.step % 5000 == 0:
                self.eval()

            total_loss = 0
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                # batch: actions(256,8,6) + conditions({0:(256,1,17)})
                loss, infos = self.model.loss(*batch)

                loss = loss / self.gradient_accumulate_every
                loss.backward()

                total_loss += loss.detach().item()

                wandb.define_metric("Train/diffusion loss", step_metric='custom_step')
                wb_log_dict = {
                    'Train/diffusion loss': infos['diffusion_noise_loss'].detach().item(),
                    "custom_step": self.step
                }
                wandb.log(wb_log_dict)
                # self.writer.add_scalar('Train/diffusion loss', infos['diffusion_noise_loss'].detach().item(), global_step=self.step)
                # print(f'Step: {self.step}, diffusion loss: {infos["diffusion_noise_loss"].detach().item()}')

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save()

            wandb.define_metric("Train/total_loss", step_metric='custom_step')
            wb_log_dict = {
                'Train/total_loss': total_loss,
                "custom_step": self.step
            }
            wandb.log(wb_log_dict)
            # self.writer.add_scalar('Train/total_loss', total_loss, global_step=self.step)

            self.step += 1

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            # 'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.bucket, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        temp_path = os.path.join(savepath, f'temp_{self.ckpt_name}')
        savepath = os.path.join(savepath, f'{self.ckpt_name}')

        torch.save(data, temp_path)
        os.replace(temp_path, savepath)
        # logger.print(f'[ utils/training ] Saved model to {savepath}')

    def load(self):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.bucket, f'checkpoint/{self.ckpt_name}')
        # data = logger.load_torch(loadpath)
        # data = torch.load(loadpath)
        data = torch.load(loadpath, map_location=self.device)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

        logger.print(f'[ utils/training ] Loaded model from {loadpath}')

    def eval(self):
        self.model.eval()
        # task_metaworld = ['door-close-v2', 'plate-slide-v2', 'bin-picking-v2', 'door-unlock-v2', 'hand-insert-v2', 'box-close-v2', 'door-lock-v2']
        ismt10 = (self.dataset.env_name == 'metaworld10')
        iseasy = (self.dataset.env_name == 'metaworldeasy')
        ismed = (self.dataset.env_name == 'metaworldmedium')
        ishard = (self.dataset.env_name == 'metaworldhard')
        isvhard = (self.dataset.env_name == 'metaworldveryhard')

        if ismt10:
            task_metaworld = ['basketball-v2', 'peg-insert-side-v2', 'drawer-open-v2']
        elif iseasy:
            task_metaworld = ['button-press-topdown-wall-v2', 'door-open-v2', 'handle-press-side-v2', 'plate-slide-back-side-v2']
        elif ismed:
            task_metaworld = ['basketball-v2', 'bin-picking-v2', 'box-close-v2', 'hammer-v2','peg-insert-side-v2', 'push-wall-v2']
        elif ishard:
            task_metaworld = ['assembly-v2', 'hand-insert-v2', 'pick-out-of-hole-v2', 'pick-place-v2', 'push-v2', 'push-back-v2']
        elif isvhard:
            task_metaworld = ['shelf-place-v2', 'disassemble-v2', 'stick-pull-v2', 'stick-push-v2', 'pick-place-wall-v2']
        else:
            task_metaworld = [self.dataset.env_name]        # single task

        num_eval = 10

        task = [metaworld.MT1(env).train_tasks[i] for i in range(num_eval) for env in task_metaworld]
        mt1 = [metaworld.MT1(env) for i in range(num_eval) for env in task_metaworld]
        env_list = [mt1[i].train_classes[task_metaworld[i]]() for j in range(num_eval) for i in
                    range(len(task_metaworld))]
        seed = 42

        for i in range(len(env_list)):
            env_list[i].set_task(task[i])
            env_list[i].seed(seed)

        task_score = {task: 0 for task in task_metaworld}
        task_success = {task: 0 for task in task_metaworld}

        score = 0
        total_success = 0
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        done = [False for i in env_list]
        max_episode_length = 500
        obs_list = [env.reset()[0][None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)  # (50, 39)
        obs_history = []

        for j in range(max_episode_length):
            obs = self.dataset.normalizer.normalize(obs, 'observations')  # (50, 39)
            obs_history.append(obs)
            if len(obs_history) < self.To:
                obs_history += [obs] * (self.To - len(obs_history))
            if len(obs_history) > self.To:
                obs_history.pop(0)
            obs_input = np.stack(obs_history[-2:], axis=1)
            cond = {0: to_torch(obs_input, device=self.device)}
            samples, _, _ = self.model.forward_collect_traj_ddim(cond=cond,
                                                            n_sample_steps=10,  # TODO
                                                            eta=1.0,
                                                            verbose=False)  # (50,12,4)

            # action diffusion
            start = self.To - 1
            end = start + self.Ta
            for step in range(start, end):
                action = samples[:, step, :]
                action = action.detach().cpu().numpy()
                action = self.dataset.normalizer.unnormalize(action, 'actions')

                if not any(done):
                    obs_list = []
                    for i in range(len(env_list)):
                        if done[i]:
                            continue
                        next_observation, reward, _, terminal, info = env_list[i].step(action[i])
                        obs_list.append(next_observation[None])
                        episode_rewards[i] += reward
                        if info['success'] > 1e-8:
                            env_success_rate[i] = 1
                        if terminal:
                            done[i] = True
                obs = np.concatenate(obs_list, axis=0)  #(50, 39)


        for i in range(len(task_metaworld)):
            tmp = []
            tmp_suc = 0
            for j in range(num_eval):
                tmp.append(episode_rewards[i + j * len(task_metaworld)])
                tmp_suc += env_success_rate[i + j * len(task_metaworld)]
            this_score = statistics.mean(tmp)
            success = tmp_suc / num_eval
            total_success += success
            # score += this_score
            task_score[task_metaworld[i]] += this_score
            task_success[task_metaworld[i]] += success

        for task in task_metaworld:
            # self.writer.add_scalar(f'Eval/task:{task}', task_score[task], global_step=self.step)
            wandb.define_metric(f'Eval/success rate:{task}', step_metric='custom_step')
            wb_log_dict = {
                f'Eval/success rate:{task}': task_success[task],
                "custom_step": self.step
            }
            wandb.log(wb_log_dict)
            # self.writer.add_scalar(f'Eval/success rate:{task}', task_success[task], global_step=self.step)
            # print(f'Step: {self.step}, {task} Success: {task_success[task]}')

