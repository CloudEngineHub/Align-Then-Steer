import math
import os
from typing import Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb

from torch.distributions import Normal

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)
from diffuser.models.temporal import IdentityCondition


class ActionDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim,
                 n_timesteps=1000, loss_type='l2', clip_sample=False, predict_epsilon=True,
                 action_weight=1.0, loss_discount=1.0, loss_weights=None):
        super().__init__()
        self.loss_type = loss_type
        self.horizon = horizon  
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.model = model 
        self.nn_condition = IdentityCondition(dropout=0.0)
        
        self.n_timesteps = n_timesteps
        
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)  # α累乘，也就是论文中的alpha_bar （alpha_t_bar）
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])  # alpha_{t-1}_bar

        self.n_timesteps = int(n_timesteps)  # diffusion steps
        self.clip_sample = clip_sample  # 是否将降噪后得到的x限制到特定范围内
        self.predict_epsilon = predict_epsilon  # 预测噪声还是直接预测x0

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)  # alpha_t_bar
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)  # alpha_{t-1}_bar

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))  # reciprocal: 倒数
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        self.final_alpha_cumprod = self.alphas_cumprod[0]
        
        # get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)
        
        self.n_sample_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, n_timesteps)[::-1].copy().astype(np.int64))

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        # dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)
        dim_weights = torch.ones(self.action_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        # loss_weights[0, :self.action_dim] = action_weight
        # loss_weights[:, :self.action_dim] = action_weight
        return loss_weights
    
    
     # ------------------------------------------ sampling ------------------------------------------#
    def set_timesteps(self, n_sample_steps):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            n_sample_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        if n_sample_steps > self.n_timesteps:
            raise ValueError(
                f"`n_sample_steps`: {n_sample_steps} cannot be larger than `self.n_timesteps`:"
                f" {self.n_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.n_timesteps} timesteps."
            )

        self.n_sample_steps = n_sample_steps
        step_ratio = self.n_timesteps // self.n_sample_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, n_sample_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
        self.timesteps += 1
        return self.timesteps

    def _left_broadcast(self, t, shape):
        assert t.ndim <= len(shape)
        return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

    def _get_variance_logprob(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep].to(timestep.device)
        mask_a = (prev_timestep >= 0).int().to(timestep.device)
        mask_b = 1 - mask_a
        alpha_prod_t_prev = (
                self.alphas_cumprod[prev_timestep].to(timestep.device) * mask_a
                + self.final_alpha_cumprod.to(timestep.device) * mask_b
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (
                1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance
#从这一步开始，是主要的地方
    def ddim_one_step(
            self,
            model_output,
            timestep,
            sample,
            eta
    ):
        if self.n_sample_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps'"
                " after creating the scheduler"
            )

        # pylint: disable=line-too-long
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = (
                timestep - self.n_timesteps // self.n_sample_steps
        )

        # 2. compute alphas, betas
        # tensor(10,) 0.0039 1 0 0.0195 0.9961
        alpha_prod_t = self.alphas_cumprod[timestep].to(timestep.device)
        mask_a = (prev_timestep >= 0).int().to(timestep.device)
        mask_b = 1 - mask_a
        alpha_prod_t_prev = (
                self.alphas_cumprod[prev_timestep].to(timestep.device) * mask_a
                + self.final_alpha_cumprod.to(timestep.device) * mask_b
        )
        alpha_prod_t = self._left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
        alpha_prod_t_prev = self._left_broadcast(alpha_prod_t_prev, sample.shape).to(
            sample.device
        )
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.predict_epsilon:
            pred_original_sample = (
                                           sample - beta_prod_t ** (0.5) * model_output
                                   ) / alpha_prod_t ** (0.5)
        else:
            
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 4. Clip "predicted x_0"
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance_logprob(timestep, prev_timestep).to(
            dtype=sample.dtype
        )
        std_dev_t = (eta * variance ** (0.5)).to(dtype=sample.dtype)
        std_dev_t = self._left_broadcast(std_dev_t, sample.shape).to(sample.device)

        # pylint: disable=line-too-long
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (
            0.5
        ) * pred_epsilon#修改

        # pylint: disable=line-too-long
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
                alpha_prod_t_prev ** (0.5) * pred_original_sample
                + pred_sample_direction
        )

        if eta > 0:
            device = model_output.device
            variance_noise = torch.randn(model_output.shape).to(device)
            variance = std_dev_t * variance_noise
            dist = Normal(prev_sample, std_dev_t)
            prev_sample = prev_sample.detach().clone() + variance
            mid = dist.log_prob(prev_sample.detach().clone())
            log_prob = (
                dist.log_prob(prev_sample.detach().clone())
                .mean(dim=-1)
                .mean(dim=-1)
                .detach()
                .cpu()
            )
        return prev_sample, pred_original_sample, log_prob

    def ddim_step_logprob(
            self,
            model_output,
            timestep,
            sample,
            next_sample,
            eta
    ):  # pylint: disable=g-bare-generic
        if self.n_sample_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps'"
                " after creating the scheduler"
            )

        # pylint: disable=line-too-long
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = (
                timestep - self.n_timesteps // self.n_sample_steps
        )

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep].to(timestep.device)
        mask_a = (prev_timestep >= 0).int().to(timestep.device)
        mask_b = 1 - mask_a
        alpha_prod_t_prev = (
                self.alphas_cumprod[prev_timestep].to(timestep.device) * mask_a
                + self.final_alpha_cumprod.to(timestep.device) * mask_b
        )
        alpha_prod_t = self._left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
        alpha_prod_t_prev = self._left_broadcast(alpha_prod_t_prev, sample.shape).to(
            sample.device
        )
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.predict_epsilon:
            pred_original_sample = (
                                        sample - beta_prod_t ** (0.5) * model_output
                                ) / alpha_prod_t ** (0.5)
        else:
            pred_original_sample = model_output

        # 4. Clip "predicted x_0"
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance_logprob(timestep, prev_timestep).to(
            dtype=sample.dtype
        )
        std_dev_t = (eta * variance ** (0.5)).to(dtype=sample.dtype)
        std_dev_t = self._left_broadcast(std_dev_t, sample.shape).to(sample.device)

        # pylint: disable=line-too-long
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (
            0.5
        ) * model_output

        # pylint: disable=line-too-long
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
                alpha_prod_t_prev ** (0.5) * pred_original_sample
                + pred_sample_direction
        )

        if eta > 0:
            device = model_output.device
            variance_noise = torch.randn(model_output.shape).to(device)
            variance = std_dev_t * variance_noise
            dist = Normal(prev_sample, std_dev_t)  # Normal(means, sigma): 设置一个高斯分布
            mid = dist.log_prob(next_sample.detach().clone())
            log_prob = (
                dist.log_prob(next_sample.detach().clone())  # log_prob(x)用来计算输入数据x在分布中的对于概率密度的对数
                .mean(dim=-1)
                .mean(dim=-1)
            )

        return log_prob

    def forward_collect_traj_ddim(
            self,
            cond,
            n_sample_steps,
            eta,
            verbose=True,
    ):
        # pylint: disable=line-too-long
        cond = self.nn_condition(cond[0])
        device = cond.device
        batch_size = len(cond)
        shape = (batch_size, self.horizon, self.action_dim)
        # 4. Prepare timesteps
        self.set_timesteps(n_sample_steps)
        timesteps = self.timesteps

        # 5. Prepare latent variables
        latents = torch.randn(shape).to(device)

        latents_list = []
        log_prob_list = []
        latents_list.append(latents.detach().clone().cpu())

        # 7. Denoising loop
        progress = utils.Progress(self.n_sample_steps) if verbose else utils.Silent()
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents
            # predict the noise
            t = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
            noise_pred = self.model(latent_model_input, t, cond)  # AuxiliaryUNet.forward

            # now we get the predicted noise
            prev_sample, pred_original_sample, log_prob = self.ddim_one_step(noise_pred, t, latents, eta)
            latents = prev_sample
            latents_list.append(latents.detach().clone().cpu())
            log_prob_list.append(log_prob.detach().clone().cpu())
            progress.update({'t': i})

        progress.close()

        final = latents

        return final, latents_list, log_prob_list
    
    # Feed transitions pairs and old model
    def forward_calculate_logprob(
            self,
            latents,
            next_latents,
            ts,
            n_sample_steps,
            eta,
            cond,
            model_copy=None,
    ):
        # pylint: disable=line-too-long
        # 2. Define call parameters
        device = latents.device
        cond = self.nn_condition(cond)

        # 3. Prepare timesteps
        self.set_timesteps(n_sample_steps)
        timesteps = self.timesteps
        model_times = timesteps[ts].to(device)

        # 4. Prepare latent variables
        latents_list = []
        latents_list.append(latents.detach().clone())

        # 6. Denoising loop
        # for training loops:
        # with self.progress_bar(total=n_sample_steps) as progress_bar:
        # for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = latents.to(device)
        time_model_input = model_times.to(device)

        # predict the noise residual
        noise_pred = self.model(latent_model_input, time_model_input, cond)

        # noise_pred.register_hook(lambda grad: print(grad.abs().sum().item()))
        # noise_pred.register_hook(lambda grad: grad * 100)
        # self.set_grad_coefs()
        # hook_fn = make_grad_hook(self.grad_coefs[ts])
        # noise_pred.register_hook(hook_fn)

        # add regularization
        if model_copy is not None:
            old_noise_pred = model_copy(latent_model_input, time_model_input, cond)
            kl_regularizer = (noise_pred - old_noise_pred) ** 2
        else:
            kl_regularizer = torch.full_like(noise_pred, -1.0, dtype=torch.float32)

        unsqueeze2x = lambda x: x[Ellipsis, None, None]
        model_times = unsqueeze2x(model_times).to(noise_pred.device)
        log_prob = self.ddim_step_logprob(noise_pred, model_times, latents, next_latents, eta)

        return log_prob, kl_regularizer

    def get_gt_noise(self, x_start, x_t, t):
        noise = (
                (x_t - extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start) / 
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        )
        return noise
            
    # ------------------------------------------ training ------------------------------------------#

    # 前向过程
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample
    
    # x_start: real data
    # cond: current state
    # t: diffusion step
    def info_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        # 前向过程加噪
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # (256,8,6)
        x_recon = self.model(x_noisy, t, cond)  

        assert noise.shape == x_recon.shape

        # denoising matching term
        if self.predict_epsilon:
            # LOSS：预测噪声与真实噪声的MSE
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        info['diffusion_noise_loss'] = loss

        return loss, info

    # x: real data
    # cond: current state
    def loss(self, x, cond):
        batch_size = len(x) 
        # 随机生成diffusion step
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()  # tensor(128, )
        
        cond = self.nn_condition(cond[0])

        # 计算infodiff loss
        # x: (256,8,6) cond: (256,1,17)  t: (256,)
        diffusion_loss, info = self.info_losses(x, cond, t)

        return diffusion_loss, info
