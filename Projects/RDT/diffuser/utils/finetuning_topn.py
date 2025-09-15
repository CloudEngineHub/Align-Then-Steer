from bisect import insort
import datetime
import os
import copy
import random
import numpy as np
import torch
# import wandb
import pathlib
from typing import Optional, Dict
from .arrays import to_torch
from .timer import Timer
from ml_logger import logger
import metaworld
from .training import EMA
from .. import utils
from ..models.temporal import ValueMulti
from diffuser.utils.video import VideoRecorder, plot_grip_trajs
import torch.nn.functional as F
from torch.optim import lr_scheduler
import wandb
# from PTrm import TransRewardModel, HLGaussLoss


import matplotlib.pyplot as plt


def init_or_resume_wandb_run(wandb_id_file_path: pathlib.Path,
                             project_name: Optional[str] = None,
                             run_name: Optional[str] = None,
                             dir: Optional[str] = None,
                             config: Optional[Dict] = None,
                             write_flag=True):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file.

        Returns the config, if it's not None it will also update it first

       NOTE:
           Make sure that wandb_id_file_path.parent exists before calling this function
    """
    # if the run_id was previously saved, resume from there
    if wandb_id_file_path.exists():
        resume_id = wandb_id_file_path.read_text()
        wandb.init(project=project_name,
                   name=run_name,
                   resume=resume_id,
                   dir=dir,
                   config=config)
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        run = wandb.init(project=project_name, name=run_name, dir=dir, config=config)
        if write_flag:
            wandb_id_file_path.write_text(str(run.id))

    wandb_config = wandb.config
    if config is not None:
        # update the current passed in config with the wandb_config
        config.update(wandb_config)
    return config


def get_random_indices(num_indices, sample_size):
    """Returns a random sample of indices from a larger list of indices.

  Args:
      num_indices (int): The total number of indices to choose from.
      sample_size (int): The number of indices to choose.

  Returns:
      A numpy array of `sample_size` randomly chosen indices.
  """
    return np.random.choice(num_indices, size=sample_size, replace=False)


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class Finetuner(object):
    def __init__(
            self,
            pretrained_diffusion_model,
            dataset,
            task='door-close-v2',
            seed=100,
            finetune_steps=1500010,
            horizon=100,
            To=2,
            Ta=8,
            eval_freq=2000,
            skill_choose_num=50,
            finetune_lr=1e-5,
            weight_decay=1e-2,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            gradient_accumulate_every=2,
            buffer_size=10000,
            save_freq=100,
            num_eval_episodes=10,
            p_step=5,
            p_batch_size=32,
            ratio_clip=1e-4,
            reward_weight=10,
            kl_weight=0.0,
            kl_outer_weight=0.0,
            bc_weight=0.0,
            bc_outer_weight=0.0,
            kl_warmup=-1,
            lr_exp=0,
            eta=1.0,
            n_sample_steps=100,
            max_grad=0.5,
            init_episode=10,
            top_n=10,
            expert_buffersize=1,
            use_expert=False,
            skill_index=None,
            pretrain_bucket=None,
            use_ema=False,
            save_model=False,
            ckpt_name=None,
            finetune_bucket=None,
            log_dir=None,
            finetune_device='cuda',
    ):
        super().__init__()
        wandb_id_file_path = pathlib.Path(
            f"/checkpoint/chenjia/infodec_check/wandb/wandb_txt/MT50FT_topn_{task}_{seed}_{bc_outer_weight}_{reward_weight}_{init_episode}_{p_step}_wandb_runid.txt")
        wandb_config = init_or_resume_wandb_run(wandb_id_file_path, project_name="MT50_FT",
                                                run_name=f'{task}_topn_seed{seed}_bc{bc_outer_weight}_rew{reward_weight}_init{init_episode}',
                                                dir='/checkpoint/chenjia/infodec_check/wandb', write_flag=save_model)
        # wandb.init(wandb_config)
        wandb.config.update(
            dict(task=task, seed=seed, bc_outer_weight=bc_outer_weight, reward_weight=reward_weight,
                 init_episode=init_episode,
                 p_step=p_step, finetune_lr=finetune_lr, lr_exp=lr_exp))
        wandb.define_metric("custom_step")
        wandb.define_metric("gradient_step")

        self.device = finetune_device

        self.dataset = dataset
        self.observation_dim = self.dataset.observation_dim
        self.action_dim = self.dataset.action_dim

        self.model = pretrained_diffusion_model

        self.save_freq = save_freq
        self.gradient_accumulate_every = gradient_accumulate_every
        self.task = task
        self.seed = seed

        # TODO: seed or not?
        train_mt1 = metaworld.MT1(task, seed=seed)
        # train_mt1 = metaworld.MT1(task)

        self.env_tasks = train_mt1.train_tasks
        self.train_env = train_mt1.train_classes[task]()
        self.train_env.set_task(random.choice(self.env_tasks))
        self.train_env.seed(seed)

        self.finetune_steps = finetune_steps
        self.horizon = horizon
        self.To = To
        self.Ta = Ta
        self.eval_freq = eval_freq
        self.accumlated_reward = 0
        self.episode_step, self.episode_reward = 0, 0
        self.global_step = 0
        self.gradient_step = 0
        self.global_episode = 0
        self.skill_choose_num = skill_choose_num
        self.skill = None
        self.skill_index = skill_index
        self.num_eval_episodes = num_eval_episodes
        self.eta = eta
        self.n_sample_steps = n_sample_steps

        self.buffer_size = buffer_size
        self.top_n = top_n
        self.expert_buffersize = expert_buffersize
        self.success_idx = []

        self.p_step = p_step
        self.p_batch_size = p_batch_size
        self.ratio_clip = ratio_clip
        self.reward_weight = reward_weight
        self.kl_weight = kl_weight
        self.kl_outer_weight = kl_outer_weight
        self.kl_warmup = kl_warmup
        self.bc_weight = bc_weight
        self.bc_outer_weight = bc_outer_weight
        self.max_grad = max_grad
        self.init_episode = init_episode
        self.success_episode = 0

        self.tot_kl = 0
        self.tot_bc = 0
        self.tot_p_loss = 0
        self.finetune_lr = finetune_lr
        self.rm_mean = 0
        self.rm_std = 0
        self.buffer_max = 0
        self.buffer_min = 0
        self.best_eval_rew = 0

        self.use_expert = use_expert
        self.save_model = save_model

        self.pretrain_bucket = pretrain_bucket
        self.finetune_bucket = finetune_bucket
        self.log_dir = log_dir

        log_dir_parts = log_dir.split('/')
        prefix = log_dir_parts[-1]

        # self.video_dir = '/ailab/user/baichenjia/fcy/DPOK/videos/' + prefix
        # if not os.path.exists(self.video_dir):
        #     os.makedirs(self.video_dir)

        self.base_rate = 0
        self.lr_exp = lr_exp

        self.ckpt_name = ckpt_name
        self.use_ema = use_ema
        pretrain_loadpath = os.path.join(self.pretrain_bucket, f'checkpoint/{self.ckpt_name}')
        if os.path.exists(pretrain_loadpath):
            self.load_pretrain()
        else:
            raise FileNotFoundError(f'Pretrain model path: {pretrain_loadpath} does not exist')

        self.finetune_loadpath = os.path.join(self.finetune_bucket, f'checkpoint/{self.task}_{self.seed}.pt')
        if os.path.exists(self.finetune_loadpath):
            self.load_finetune_checkpoint()
        else:
            logger.print(f'Finetune model path: {self.finetune_loadpath} does not exist, start from scratch')

        self.model.requires_grad_(True)
        # self.model.encoder.requires_grad_(False)
        # self.model.vq.requires_grad_(False)
        # self.model.inv_model.requires_grad_(False)

        if self.kl_weight > 0 or self.kl_outer_weight > 0:
            self.pretrained_model = copy.deepcopy(self.model.model)
            self.pretrained_model.requires_grad_(False)
        else:
            self.pretrained_model = None

        # self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=finetune_lr,
        #                                    weight_decay=weight_decay, betas=(beta1, beta2), eps=eps)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=finetune_lr)  # ,
        # weight_decay=weight_decay, betas=(beta1, beta2), eps=eps)
        self.lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=64, T_mult=2, eta_min=1e-8)

    def trim_buffer(self, state_dict, size=None, isexpert=False):
        """Delete old samples from the bufffer."""
        if size is None:
            trim_size = self.buffer_size
        else:
            trim_size = size
        if state_dict["state"].shape[0] > trim_size:
            state_dict["state"] = state_dict["state"][-trim_size:]
            state_dict["next_state"] = state_dict["next_state"][-trim_size:]
            state_dict["timestep"] = state_dict["timestep"][-trim_size:]
            state_dict["final_reward"] = state_dict["final_reward"][-trim_size:]
            state_dict["log_prob"] = state_dict["log_prob"][-trim_size:]
            state_dict["x0"] = state_dict["x0"][-trim_size:]
            state_dict["cond"] = state_dict["cond"][-trim_size:]
            if isexpert:
                state_dict["ep_reward"] = state_dict["ep_reward"][-trim_size:]

    def train_policy_func(self, state_dict, expert_dict, count, policy_steps):
        """Trains the policy function."""
        with torch.no_grad():
            indices = get_random_indices(
                state_dict["state"].shape[0], self.p_batch_size  # batch size for policy update per gpu
            )
            batch_state = state_dict["state"][indices].to(self.device)
            batch_next_state = state_dict["next_state"][indices].to(self.device)
            batch_timestep = state_dict["timestep"][indices]
            batch_final_reward = state_dict["final_reward"][indices].to(self.device)
            batch_log_prob = state_dict["log_prob"][indices]
            batch_x0 = state_dict["x0"][indices].to(self.device)
            batch_cond = state_dict["cond"][indices].to(self.device)

            indices_expert = get_random_indices(
                expert_dict["state"].shape[0], self.p_batch_size  # batch size for policy update per gpu
            )
            expert_state = expert_dict["state"][indices_expert].to(self.device)
            expert_timestep = expert_dict["timestep"][indices_expert]
            expert_x0 = expert_dict["x0"][indices_expert].to(self.device)
            expert_cond = expert_dict["cond"][indices_expert].to(self.device)

            if self.use_expert:
                bc_state = expert_state
                bc_timestep = expert_timestep
                bc_x0 = expert_x0
                bc_cond = expert_cond
            else:
                bc_state = batch_state
                bc_timestep = batch_timestep
                bc_x0 = batch_x0
                bc_cond = batch_cond

        log_prob, kl_regularizer = self.model.forward_calculate_logprob(
            # Xt-1是log_p的一个instance, 用Xt-1和Xt还原出p_theta(Xt-1|Xt,z)
            latents=batch_state,
            next_latents=batch_next_state,
            ts=batch_timestep,
            model_copy=self.pretrained_model,
            eta=self.eta,
            n_sample_steps=self.n_sample_steps,
            cond=batch_cond
        )

        # reward normalization
        # batch_final_reward = (batch_final_reward - self.rm_mean) / self.rm_std
        batch_final_reward = (batch_final_reward - self.buffer_min) / (self.buffer_max - self.buffer_min)

        if self.bc_weight > 0 or self.bc_outer_weight > 0:
            model_time = self.model.timesteps[bc_timestep].to(self.device)
            gt_noise = self.model.get_gt_noise(bc_x0, bc_state, model_time)
            pred_noise = self.model.model(bc_state, model_time, bc_cond)
            bc_regularizer = F.mse_loss(pred_noise, gt_noise, reduction='none').mean(-1).mean(-1)
        else:
            bc_regularizer = 0

        if self.kl_weight > 0 or self.kl_outer_weight > 0:
            kl_regularizer = kl_regularizer.mean(-1).mean(-1)
        else:
            kl_regularizer = 0

        adv = (self.reward_weight * batch_final_reward -
               self.kl_weight * kl_regularizer -
               self.bc_weight * bc_regularizer).to(self.device).reshape([self.p_batch_size, 1])

        # ratio = torch.exp(log_prob)
        ratio = torch.exp(log_prob - batch_log_prob.to(self.device))

        unclipped_loss = -1 * adv * ratio.reshape(
            [self.p_batch_size, 1])
        clipped_loss = -1 * adv * torch.clamp(
            ratio.reshape([self.p_batch_size, 1]),
            1.0 - self.ratio_clip,
            1.0 + self.ratio_clip,
        )

        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        # loss = torch.mean(unclipped_loss)

        if self.kl_outer_weight > 0:
            loss += self.kl_outer_weight * kl_regularizer.mean()
        if self.bc_outer_weight > 0:
            loss += self.bc_outer_weight * bc_regularizer.mean()

        loss = loss / self.gradient_accumulate_every
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad, norm_type=2)  # 3e-4

        # logging
        if self.bc_weight > 0 or self.bc_outer_weight > 0:
            self.tot_bc += bc_regularizer.mean().item() / policy_steps
        if self.kl_weight > 0 or self.kl_outer_weight > 0:
            self.tot_kl += kl_regularizer.mean().item() / policy_steps
        self.tot_p_loss += loss.item() / policy_steps

    def eval(self):
        # logger.print('Start evaluation...')
        self.model.eval()
        num_eval = 50

        tasks = [metaworld.MT1(self.task, seed=self.seed).train_tasks[i] for i in range(num_eval)]
        mt1 = [metaworld.MT1(self.task, seed=self.seed) for i in range(num_eval)]
        # tasks = [metaworld.MT1(self.task).train_tasks[i] for i in range(num_eval)]
        # mt1 = [metaworld.MT1(self.task) for i in range(num_eval)]

        env_list = [mt1[i].train_classes[self.task](render_mode="rgb_array", camera_name='corner3') for i in
                    range(num_eval)]
        seed = self.seed

        for i in range(len(env_list)):
            env_list[i].set_task(tasks[i])
            env_list[i].seed(seed)

        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        done = [False for i in env_list]
        max_episode_length = 500
        obs_list = [env.reset()[0][None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        obs_history = []

        while not any(done):
            obs = self.dataset.normalizer.normalize(obs, 'observations')  # (50. 39)
            obs_history.append(obs)
            if len(obs_history) < self.To:
                obs_history += [obs] * (self.To - len(obs_history))
            if len(obs_history) > self.To:
                obs_history.pop(0)
            obs_input = np.stack(obs_history[-2:], axis=1)
            cond = {0: to_torch(obs_input, device=self.device)}
            # cond = {0: to_torch(obs, device=self.device)[:, None, :]}
            samples, _, _ = self.model.forward_collect_traj_ddim(cond=cond,
                                                                 n_sample_steps=10,  # TODO
                                                                 eta=1.0,
                                                                 verbose=False)  # (50,8,4)

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
                obs = np.concatenate(obs_list, axis=0)

        avg_score = np.mean(episode_rewards)
        avg_success = np.sum(env_success_rate) / num_eval

        wandb.define_metric("Eval/average episode reward", step_metric='custom_step')
        wandb.define_metric("Eval/success rate", step_metric='custom_step')

        wb_log_dict = {
            'Eval/average episode reward': avg_score,
            'Eval/success rate': avg_success,
            "custom_step": self.global_step
        }
        wandb.log(wb_log_dict)

        # self.writer.add_scalar('Eval/average episode reward', avg_score, global_step=self.global_step)
        # self.writer.add_scalar('Eval/success rate', avg_success, global_step=self.global_step)
        # if avg_success > 0.66:
        #     self.save(suffix=f'{avg_success:.2f}')

        if self.global_step == 0:
            self.base_rate = avg_success
            self.best_eval_rew = avg_score
        if self.save_model:
            # if avg_score > self.best_eval_rew:
            #     self.save(isbest=True)
            #     self.best_eval_rew = avg_score
            if self.global_episode % 200 == 0:
                self.save()
        self.model.train()
        new_learning_rate = self.finetune_lr * torch.exp(
            self.lr_exp * torch.clip(torch.tensor(avg_success) - self.base_rate, min=0))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate.item()

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#
    def finetune(self):
        state_dict = {"state": torch.FloatTensor(), "next_state": torch.FloatTensor(),
                      "timestep": torch.LongTensor(), "final_reward": torch.FloatTensor(),
                      "log_prob": torch.FloatTensor(), "x0": torch.FloatTensor(), "cond": torch.FloatTensor()}
        temp_dict = {"state": torch.FloatTensor(), "next_state": torch.FloatTensor(),
                     "timestep": torch.LongTensor(), "final_reward": torch.FloatTensor(),
                     "log_prob": torch.FloatTensor(), "x0": torch.FloatTensor(), "cond": torch.FloatTensor()}
        expert_dict = {"state": torch.FloatTensor(), "next_state": torch.FloatTensor(),
                       "timestep": torch.LongTensor(), "final_reward": torch.FloatTensor(),
                       "log_prob": torch.FloatTensor(), "x0": torch.FloatTensor(), "ep_reward": torch.FloatTensor(),
                       "cond": torch.FloatTensor()}
        buffer_path = os.path.join(self.finetune_bucket, f'buffer/{self.task}_{self.seed}.pt')
        if os.path.exists(buffer_path):
            buffer = torch.load(buffer_path, map_location='cpu')
            state_dict = buffer['state_dict']
            expert_dict = buffer['expert_dict']
            logger.print(f'[ utils/training ] Recovered buffer from {buffer_path}')

        if self.global_step == 0:
            self.eval()
        policy_steps = self.gradient_accumulate_every * self.p_step
        obs = self.train_env.reset()[0][None]
        obs_history = []
        done = False
        success = False
        flag = True
        env_success_rate = []
        success_eval_freq = 20
        # while True:
        while self.global_step < self.finetune_steps:
            if done:
                self.global_episode += 1
                if len(env_success_rate) == success_eval_freq:
                    success_rate = sum(env_success_rate) / success_eval_freq

                    wandb.define_metric("Training/success_rate", step_metric='custom_step')
                    wb_log_dict = {
                        'Training/success_rate': success_rate,
                        "custom_step": self.global_step
                    }
                    wandb.log(wb_log_dict)
                    # self.writer.add_scalar('Training/success_rate', success_rate, global_step=self.global_step)

                    env_success_rate = []

                wandb.define_metric("Training/average_episode_reward", step_metric='custom_step')
                wandb.define_metric("Training/bc", step_metric='custom_step')
                wb_log_dict = {
                    'Training/average_episode_reward': self.episode_reward,
                    'Training/bc': self.tot_bc,
                    "custom_step": self.global_step
                }
                wandb.log(wb_log_dict)
                # self.writer.add_scalar('Training/average_episode_reward', self.episode_reward, global_step=self.global_step)
                # # self.writer.add_scalar('Training/kl', self.tot_kl, global_step=self.global_step)
                # self.writer.add_scalar('Training/bc', self.tot_bc, global_step=self.global_step)

                # reset env
                self.train_env.set_task(random.choice(self.env_tasks))
                obs = self.train_env.reset()[0][None]
                done = False
                success = False
                self.episode_step = 0
                self.episode_reward = 0

                if self.success_episode > self.init_episode and self.global_episode % self.eval_freq == 0:
                    # self.save()
                    self.eval()

            with torch.no_grad():
                obs = self.dataset.normalizer.normalize(obs, 'observations')
                obs_history.append(obs)
                if len(obs_history) < self.To:
                    obs_history += [obs] * (self.To - len(obs_history))
                if len(obs_history) > self.To:
                    obs_history.pop(0)
                obs_input = np.stack(obs_history[-2:], axis=1)
                cond = {0: to_torch(obs_input, device=self.device)}
                samples, latents_list, log_prob_list = self.model.forward_collect_traj_ddim(cond=cond,
                                                                                            n_sample_steps=self.n_sample_steps,
                                                                                            eta=self.eta,
                                                                                            verbose=False)

                self.accumlated_reward = 0
                Ta_cnt = 0
                start = self.To - 1
                end = start + self.Ta
                for step in range(start, end):
                    action = samples[:, step, :]
                    action = action.detach().cpu().numpy()
                    action = self.dataset.normalizer.unnormalize(action, 'actions')
                    action = action.squeeze()
                    obs, reward, _, done, info = self.train_env.step(action)
                    self.episode_reward += reward
                    self.episode_step += 1
                    self.global_step += 1
                    if done:
                        if info['success'] > 1e-8:
                            env_success_rate.append(1)
                            success = True
                        else:
                            env_success_rate.append(0)
                        flag = False
                        break

                    self.accumlated_reward += np.power(1.0, Ta_cnt) * reward
                    # self.accumlated_reward = reward
                    flag = True
                    Ta_cnt += 1

            if flag:
                for i in range(len(latents_list) - 1):
                    temp_dict["state"] = torch.cat((temp_dict["state"], latents_list[i]))
                    temp_dict["next_state"] = torch.cat(
                        (temp_dict["next_state"], latents_list[i + 1])
                    )
                    temp_dict["timestep"] = torch.cat(
                        (temp_dict["timestep"], torch.LongTensor([i]))
                    )
                    temp_dict["final_reward"] = torch.cat(
                        (temp_dict["final_reward"], torch.tensor(self.accumlated_reward)[None])  # r(X0)   for finetuing
                    )
                    temp_dict["log_prob"] = torch.cat(
                        (temp_dict["log_prob"], log_prob_list[i])
                    )
                    temp_dict["x0"] = torch.cat(
                        (temp_dict["x0"], samples.cpu())
                    )
                    temp_dict["cond"] = torch.cat(
                        (temp_dict["cond"], cond[0].cpu())
                    )

            if done:
                expert_dict["state"] = torch.cat((expert_dict["state"], temp_dict["state"]))
                expert_dict["next_state"] = torch.cat(
                    (expert_dict["next_state"], temp_dict["next_state"])
                )
                expert_dict["timestep"] = torch.cat(
                    (expert_dict["timestep"], temp_dict["timestep"])
                )
                expert_dict["final_reward"] = torch.cat(
                    (expert_dict["final_reward"], temp_dict["final_reward"])
                    # r(X0)   for finetuing
                )
                expert_dict["log_prob"] = torch.cat(
                    (expert_dict["log_prob"], temp_dict["log_prob"])
                )
                expert_dict["x0"] = torch.cat(
                    (expert_dict["x0"], temp_dict["x0"])
                )
                expert_dict["ep_reward"] = torch.cat(
                    (expert_dict["ep_reward"], torch.full_like(temp_dict["final_reward"], self.episode_reward))
                )
                expert_dict["cond"] = torch.cat(
                    (expert_dict["cond"], temp_dict["cond"])
                )

                ep_rewards = expert_dict["ep_reward"]
                sorted_indices = torch.argsort(ep_rewards)
                for key in expert_dict.keys():
                    expert_dict[key] = expert_dict[key][sorted_indices]  # 把整个dict按reward排序

                self.trim_buffer(expert_dict, size=620 * self.top_n, isexpert=True)  # 把低reward的去除

                if self.global_episode > 5 * self.top_n:
                    self.success_episode += 1

            # if (done and success):
            if self.success_episode < self.init_episode:
                buffer_flag = False
            else:
                buffer_flag = done
            if buffer_flag:
                if state_dict["state"].numel() == 0:
                    state_dict["state"] = expert_dict["state"]
                    state_dict["next_state"] = expert_dict["next_state"]
                    state_dict["timestep"] = expert_dict["timestep"]
                    state_dict["final_reward"] = expert_dict["final_reward"]
                    state_dict["log_prob"] = expert_dict["log_prob"]
                    state_dict["x0"] = expert_dict["x0"]
                    state_dict["cond"] = expert_dict["cond"]

                state_dict["state"] = torch.cat((state_dict["state"], temp_dict["state"]))
                state_dict["next_state"] = torch.cat(
                    (state_dict["next_state"], temp_dict["next_state"])
                )
                state_dict["timestep"] = torch.cat(
                    (state_dict["timestep"], temp_dict["timestep"])
                )
                state_dict["final_reward"] = torch.cat(
                    (state_dict["final_reward"], temp_dict["final_reward"])
                    # r(X0)   for finetuing
                )
                state_dict["log_prob"] = torch.cat(
                    (state_dict["log_prob"], temp_dict["log_prob"])
                )
                state_dict["x0"] = torch.cat(
                    (state_dict["x0"], temp_dict["x0"])
                )
                state_dict["cond"] = torch.cat(
                    (state_dict["cond"], temp_dict["cond"])
                )
                self.trim_buffer(state_dict)

            if self.success_episode < self.init_episode:
                train_flag = False
            else:
                train_flag = done
            if train_flag:
                final_reward_np = state_dict["final_reward"].cpu().numpy()
                final_reward_unique = np.unique(final_reward_np)
                self.rm_std = torch.tensor(np.std(final_reward_unique))
                self.rm_mean = torch.tensor(np.mean(final_reward_unique))
                self.buffer_max = torch.tensor(np.max(final_reward_unique))
                self.buffer_min = torch.tensor(np.min(final_reward_unique))

                self.model.train()
                self.tot_kl, self.tot_p_loss, self.tot_bc = 0, 0, 0
                for _ in range(self.p_step):
                    self.optimizer.zero_grad()
                    for accum_step in range(self.gradient_accumulate_every):
                        self.train_policy_func(state_dict, expert_dict, self.global_step, policy_steps)

                    # 计算总体梯度
                    total_gradient = 0.0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            total_gradient += param.grad.abs().sum().item()

                    wandb.define_metric("Gradient/gradient", step_metric='gradient_step')
                    wb_log_dict = {
                        'Gradient/gradient': total_gradient,
                        "gradient_step": self.gradient_step
                    }
                    wandb.log(wb_log_dict)
                    # self.writer.add_scalar('Gradient/gradient', total_gradient, global_step=self.gradient_step)

                    self.optimizer.step()
                    # self.lr_scheduler.step()

                    self.gradient_step += 1

            obs = obs[None]
            if done:
                temp_dict = {"state": torch.FloatTensor(), "next_state": torch.FloatTensor(),
                             "timestep": torch.LongTensor(), "final_reward": torch.FloatTensor(),
                             "log_prob": torch.FloatTensor(), "x0": torch.FloatTensor(), "cond": torch.FloatTensor()}
                if self.save_model and self.success_episode > self.init_episode and self.global_episode % 80 == 0:
                    self.save()
                    self.save_buffer(state_dict, expert_dict)

    def save(self, suffix=None):
        '''
            saves model to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.global_step,
            'episode': self.global_episode,
            'success_episode': self.success_episode,
            'gradient_step': self.gradient_step,
            'model': self.model.state_dict(),
        }
        if suffix is not None:
            savepath = os.path.join(self.finetune_bucket, f'checkpoint/{self.task}_{self.seed}_{suffix}.pt')
        else:
            savepath = os.path.join(self.finetune_bucket, f'checkpoint/{self.task}_{self.seed}.pt')
        # os.makedirs(savepath, exist_ok=True)

        temp_path = os.path.join(self.finetune_bucket, f'checkpoint/{self.task}_{self.seed}_temp.pt')

        torch.save(data, temp_path)
        os.replace(temp_path, savepath)
        logger.print(f'[ utils/training ] Saved finetune model to {savepath}')

    def save_buffer(self, state_dict, expert_dict):
        '''
            saves model to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'state_dict': state_dict,
            'expert_dict': expert_dict,
        }
        savepath = os.path.join(self.finetune_bucket, f'buffer/{self.task}_{self.seed}.pt')
        # os.makedirs(savepath, exist_ok=True)

        temp_path = os.path.join(self.finetune_bucket, f'buffer/{self.task}_{self.seed}_temp.pt')

        torch.save(data, temp_path)
        os.replace(temp_path, savepath)
        logger.print(f'[ utils/training ] Saved buffer to {savepath}')

    def load_finetune_checkpoint(self):
        '''
            loads model from disk
        '''
        loadpath = os.path.join(self.finetune_bucket, f'checkpoint/{self.task}_{self.seed}.pt')
        data = torch.load(loadpath, map_location=self.device)

        self.global_step = data['step']
        self.model.load_state_dict(data['model'])
        self.global_episode = data['episode']
        self.success_episode = data['success_episode']
        self.gradient_step = data['gradient_step']

        logger.print(f'[ utils/training ] Recovered finetune model from {loadpath}')

    def load_pretrain(self):
        '''
            loads model from disk
        '''
        loadpath = os.path.join(self.pretrain_bucket, f'checkpoint/{self.ckpt_name}')
        data = torch.load(loadpath, map_location=self.device)
        if self.use_ema:
            self.model.load_state_dict(data['ema'])
        else:
            self.model.load_state_dict(data['model'])
        logger.print(f'[ utils/training ] Loaded pretrain model from {loadpath}')
