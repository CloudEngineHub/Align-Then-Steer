from tkinter import NO
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import wandb
from diffusion_policy.model.vae.info_vae import InfoVAE


## Explicit Classifier Guidance over latent space
class LatentGuidence(nn.Module):
    def __init__(self, vae_config):
        super().__init__()
        self.vae_model = InfoVAE(latent_dim=[1, vae_config['latent_dim']], dropout=0.0, nfeats=vae_config['input_channels'])
        checkpoint: str = torch.load(vae_config['pretrained_path'])
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        self.vae_model.load_state_dict(checkpoint)

        if vae_config['mask'] is not None:
            self.imask = np.array([i for i in range(128) if i not in vae_config['mask']])
        else:
            self.imask = None

        self.lambda_: float = vae_config["lambda"]
        self.clamp :float = vae_config.get("clamp", 0.1)

    def forward(self, alpha_t_bar, pred, a0, a_t, pred_type, reduction):
        grad_coef = 1.
        if pred_type == 'epsilon':
            grad_coef *= - ((1 - alpha_t_bar) ** 0.5)
        elif pred_type == 'sample':
            grad_coef *= (( 1 - alpha_t_bar) / ( alpha_t_bar ** 0.5 ))
        else:
            raise ValueError(f'Currently we donot support pred type: {pred_type}.')


        with torch.enable_grad():
            with torch.autograd.set_detect_anomaly(True):
                a_t = a_t.detach()
                a_t.requires_grad_()
                alpha_t_bar = alpha_t_bar.view(-1, 1, 1)
                pred_z, pred_dist, pred_mu, pred_std  = self.vae_model.encode(a_t)
                target_z, target_dist, target_mu, target_std = self.vae_model.encode(a0)
                eps = torch.randn(pred_mu.shape).to(pred_mu.device)

                pred_latent = pred_mu + pred_std * eps
                target_latent = target_mu + target_std * eps
                
                y = F.mse_loss(pred_latent, target_latent)* (-1.)

                grad = torch.autograd.grad([y], [a_t])[0]

                if self.imask is not None:
                    grad[:, :, self.imask] = 0
                
                grad_norm = torch.norm(grad, p=2)
                grad *= grad_coef

                if wandb.run is not None:
                    wandb.log({"grad_norm": grad_norm.item()})
                grad = torch.clamp(grad, min=-1 * self.clamp, max=self.clamp)
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    print("[Error] grad has NaN or Inf values")
                    
                guide = pred + self.lambda_ * grad
                loss = F.mse_loss(guide, a0, reduction=reduction)
                if torch.isnan(loss).any():
                    print("Guided term NaN or Inf")
                    print("lambda:", self.lambda_)
                    print("grad:", grad.min().item(), grad.max().item())
                    loss = torch.zeros((1), dtype=torch.bfloat16, device=loss.device)
            return loss