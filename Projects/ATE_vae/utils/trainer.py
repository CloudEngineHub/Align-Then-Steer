# Description: This file contains the class definition for the trainer class.
import os

import torch
import wandb
import copy
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
from termcolor import cprint

from utils.Dataset import RDTSeqDataset
from models.info_vae import InfoVAE, Mldloss
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import numpy as np
from omegaconf import OmegaConf
import shutil

def add_noise(noise_scheduler, action, batch_size=64):
        noise = torch.randn(
        action.shape, dtype=action.dtype, device=action.device
                )
        # Sample random diffusion timesteps
        timesteps = torch.randint(
                0, 3, 
                (batch_size,), device=action.device
        ).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = noise_scheduler.add_noise(
                action, noise, timesteps)
        return noisy_action

def save_tensor_or_array(tensor_or_array, file_path):
    if isinstance(tensor_or_array, torch.Tensor):
        np.savetxt(file_path, tensor_or_array.detach().cpu().numpy(), fmt="%.6f")
    elif isinstance(tensor_or_array, np.ndarray):
        np.savetxt(file_path, tensor_or_array, fmt="%.6f")
    else:
        raise TypeError(f"Unsupported type: {type(tensor_or_array)}")

class trainer:
    def __init__(self, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg

        self.traing_dict = OmegaConf.load(os.path.join(self.cfg.pretrained_model_dir, "training_config.yaml")) \
                    if self.cfg.enable_resume else {}

        if self.cfg.enable_resume:
            cprint(f'{"*"*40}\n Resuming Training! \n{"*"*40}\n', "blue")
            self.resume_model()
        else:
            self.time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.traing_dict["time"] = self.time
            self.switch_stage(self.cfg.start_stage)
            self.init_model()
            if self.traing_dict["stage"] == "Stage2":
                self.loaded_mu  = torch.tensor(np.loadtxt(os.path.join(self.cfg.stage1_dataset_dir,"mu.txt"),  dtype=np.float32),
                                               dtype=torch.float32).unsqueeze(0).expand(self.cfg.batch_size, self.cfg.latent_dim).to(self.device)
                self.loaded_std = torch.tensor(np.loadtxt(os.path.join(self.cfg.stage1_dataset_dir,"std.txt"), dtype=np.float32),
                                               dtype=torch.float32).unsqueeze(0).expand(self.cfg.batch_size, self.cfg.latent_dim).to(self.device)
        self._make_dataloaders()

        cprint("Compiling model...", "magenta")
        torch.compile(self.model)
        cprint("Compiled...", "magenta")
        self.model.to(self.device)
        torch.set_float32_matmul_precision('high')

        if cfg.wandb_flag:
            wandb.init(project="vae-action", 
            name=f'vae-run-{self.time}',
                config={
                    "architecture": type(self.model).__name__,
                    "stage1_input_length": self.cfg.stage1_s_length,
                    "stage2_input_length": self.cfg.stage2_s_length,
                    "latent_dim": self.cfg.latent_dim,
                    "learning_rate": self.cfg.learning_rate,
                    "batch_size": self.cfg.batch_size,
                    "Stage1_epochs": self.cfg.stage1_num_epoch,
                    "Stage2_epochs": self.cfg.stage2_num_epoch,
                    "optimizer": "AdamW",
                })
    
    def switch_stage(self, stage):
        assert stage in ["Stage1", "Stage2"], "Stage must be either 'Stage1' or 'Stage2'"
        self.traing_dict["stage"] = stage
        self.traing_dict["epoch"] = 0
        self.traing_dict["BestLoss"] = float("inf")
        self.cfg.s_length = self.cfg.stage1_s_length if stage == "Stage1" else self.cfg.stage2_s_length
        self.init_model()
        self._make_dataloaders()
        cprint(f'Switched to {stage}!', "blue")
       
    def resume_model(self):
        self.time = self.traing_dict.time
        self.model = InfoVAE(latent_dim=[1, self.cfg.latent_dim], dropout=0, nfeats=self.cfg.in_channels) \
            .load_state_dict(torch.load(os.path.join(self.cfg.pretrained_model_dir, "Last.pth")))
        self.BestModel = InfoVAE(latent_dim=[1, self.cfg.latent_dim], dropout=0, nfeats=self.cfg.in_channels) \
            .load_state_dict(torch.load(os.path.join(self.cfg.pretrained_model_dir, "Best.pth"))) \
            if os.path.exists(os.path.join(self.cfg.pretrained_model_dir, "Best.pth")) else None
        
        self.optimizer = optim.AdamW(self.model.parameters(),
                            lr=self.cfg.learning_rate,
                            weight_decay=self.cfg.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                    gamma = 0.95)
        self.optimizer.load_state_dict(torch.load(os.path.join(self.cfg.pretrained_model_dir, f"optimizer.pth")))
        self.scheduler.load_state_dict(torch.load(os.path.join(self.cfg.pretrained_model_dir, f"scheduler.pth")))
        
        if self.traing_dict["stage"] == "Stage2":
            self.s_length = self.cfg.stage2_s_length
            self.loaded_mu  = torch.tensor(np.loadtxt(os.path.join(self.cfg.pretrained_model_dir,"mu.txt"),  dtype=np.float32),
                                           dtype=torch.float32).unsqueeze(0).expand(self.cfg.batch_size, self.cfg.latent_dim).to(self.device)
            self.loaded_std = torch.tensor(np.loadtxt(os.path.join(self.cfg.pretrained_model_dir,"std.txt"), dtype=np.float32),
                                           dtype=torch.float32).unsqueeze(0).expand(self.cfg.batch_size, self.cfg.latent_dim).to(self.device)
        else:
            self.s_length = self.cfg.stage1_s_length

        self.length_list = [self.cfg.s_length] * self.cfg.batch_size

    def init_model(self):
        self.BestModel = None
        self.model = InfoVAE(latent_dim=[1, self.cfg.latent_dim], dropout=0, nfeats=self.cfg.in_channels)
        self.length_list = [self.cfg.s_length] * self.cfg.batch_size
        self.scaler = torch.cuda.amp.GradScaler()
        self.optimizer = optim.AdamW(self.model.parameters(),
                                lr=self.cfg.learning_rate,
                                weight_decay=self.cfg.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                    gamma = 0.95)

    def compute_mu_and_std(self):
        Over_all_mu = np.zeros((self.cfg.batch_size, self.cfg.latent_dim))
        Over_all_std = np.zeros((self.cfg.batch_size, self.cfg.latent_dim))
        noise_scheduler = DDPMScheduler(
                        num_train_timesteps=10,
                        beta_schedule='squaredcos_cap_v2',
                        prediction_type='sample',
                        clip_sample=False,
                    )
        cnt = 0
        for action in tqdm(self.valDataloader):
                input = action.to('cuda')
                length_list = [self.cfg.s_length] * self.cfg.batch_size
                cnt += 1

                input = add_noise(noise_scheduler, input, batch_size=self.cfg.batch_size)
                *_, mu, std = self.model.encode(input, length_list)
                Over_all_mu += (np.squeeze(mu.detach().to('cpu').numpy()) - Over_all_mu) / cnt
                Over_all_std += (np.squeeze(std.detach().to('cpu').numpy()) - Over_all_std) / cnt 

        for action in tqdm(self.trainDataloader):
                input = action.to('cuda')
                length_list = [self.cfg.s_length] * self.cfg.batch_size
                cnt += 1
                input = add_noise(noise_scheduler, input, batch_size=self.cfg.batch_size)
                
                *_, mu, std = self.model.encode(input, length_list)
                Over_all_mu += (np.squeeze(mu.detach().to('cpu').numpy()) - Over_all_mu) / cnt
                Over_all_std += (np.squeeze(std.detach().to('cpu').numpy()) - Over_all_std) / cnt 

        return Over_all_mu.mean(0), Over_all_std.mean(0)

    def _make_dataloaders(self):
        if self.traing_dict.get("stage", "Stage1") == "Stage1":
            cprint(f'{"*"*40}\n Stage1 Training! \n{"*"*40}\n', "blue")
            trainDataset = RDTSeqDataset(os.path.join(self.cfg.stage1_dataset_dir, "train"), action_seq_length=self.cfg.s_length)
            self.trainDataloader = DataLoader(trainDataset, batch_size=self.cfg.batch_size,
                                        shuffle=True, num_workers=4,
                                        pin_memory=True, drop_last=True)
            valDataset = RDTSeqDataset(os.path.join(self.cfg.stage1_dataset_dir,"val"), action_seq_length=self.cfg.s_length)
            self.valDataloader = DataLoader(valDataset, batch_size=self.cfg.batch_size,
                                    shuffle=True, num_workers=4,
                                    pin_memory=True, drop_last=True)
            self.traing_dict["stage"] = "Stage1"
        
        else:
            cprint(f'{"*"*40}\n Stage2 Training! \n{"*"*40}\n', "blue")
            trainDataset = RDTSeqDataset(os.path.join(self.cfg.stage2_dataset_dir, "train"), action_seq_length=self.cfg.s_length)
            self.trainDataloader = DataLoader(trainDataset, batch_size=self.cfg.batch_size,
                                        shuffle=True, num_workers=4,
                                        pin_memory=True, drop_last=True)
            valDataset = RDTSeqDataset(os.path.join(self.cfg.stage2_dataset_dir,"val"), action_seq_length=self.cfg.s_length)
            self.valDataloader = DataLoader(valDataset, batch_size=self.cfg.batch_size,
                                    shuffle=True, num_workers=4,
                                    pin_memory=True, drop_last=True)
            
    def save(self):
        time = f'outputs/checkpoints/{self.time}/{self.traing_dict["stage"]}'
        if self.traing_dict["epoch"] == self.cfg.stage1_num_epoch - 1 or self.traing_dict["epoch"] == self.cfg.stage2_num_epoch - 1 \
            and self.traing_dict["stage"] == "Stage1":
            root = f'{time}/Last'
        else:
            root = f'{time}/{self.traing_dict["epoch"]}'
        
        os.makedirs(root, exist_ok=True)

        if len(os.listdir(time)) > self.cfg.max_save + 1:
            all_dirs = os.listdir(time)
            all_dirs.sort()
            shutil.rmtree(f'{time}/{all_dirs[0]}')

        cprint(f'BestLoss: {self.traing_dict["BestLoss"]}', on_color="on_green")
        torch.save(self.model.state_dict(),
                f'{root}/Last.pth')
        cprint(f'Saved Last Model at {root}/Last.pth', "green")
        if self.BestModel is not None:
            torch.save(self.BestModel.state_dict(),
                    f'{root}/Best.pth')
            cprint(f'Saved Best Model at {root}/Best.pth', "green")
        
        torch.save(self.optimizer.state_dict(),
                f'{root}/optimizer.pth')
        torch.save(self.scheduler.state_dict(),
                f'{root}/scheduler.pth')
        cprint(f'Saved Optimizer and Scheduler state at {root}', "green")
        OmegaConf.save(config=self.traing_dict, f=f"{root}/training_config.yaml")
        if getattr(self, "loaded_mu", None) is not None and getattr(self, "loaded_std", None) is not None:
            save_tensor_or_array(self.loaded_mu, f'{root}/mu.txt')
            save_tensor_or_array(self.loaded_std, f'{root}/std.txt')

    def _run_single_step(self, action):
        motion_z, dist_m, *_ = self.model.encode(action)
        feats_rst = self.model.decode(motion_z, self.length_list)
        
        if self.traing_dict["stage"] == "Stage2":
            if isinstance(self.loaded_mu, np.ndarray):
                self.loaded_mu = torch.tensor(self.loaded_mu, device=self.device, dtype=torch.float32)
            if isinstance(self.loaded_std, np.ndarray):
                self.loaded_std = torch.tensor(self.loaded_std, device=self.device, dtype=torch.float32)
            dist_ref = torch.distributions.Normal(self.loaded_mu, self.loaded_std)
        else:
            mu_ref = torch.zeros_like(dist_m.loc)
            scale_ref = torch.ones_like(dist_m.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        feats = {
                "m_ref": action,
                "m_rst": feats_rst,
                "dist_m": dist_m,
                "dist_ref": dist_ref,
                "z": motion_z,
            }

        loss, reconLoss, KLloss, mmd = self.LossFunc(feats)
        return loss, reconLoss, KLloss, mmd

    def _report(self, loss, reconLoss, KLloss, mmd=None):
        log_ = {
                    'loss': loss,
                    'Reconstruction_Loss':reconLoss,
                    'KLD':KLloss,
                    'learning_rate':self.optimizer.param_groups[0]['lr'],
                    'BestLoss':self.traing_dict["BestLoss"],
                    'MMD': mmd
                }

        wandb.log(log_)

    def fit(self):
        if self.traing_dict.get("stage", "Stage1") == "Stage1":
            self.LossFunc = Mldloss(False)
            for i in range(self.traing_dict["epoch"], self.cfg.stage1_num_epoch):
                self.model.train()
                logger_cnt = 0
                with tqdm(total=len(self.trainDataloader), desc=f'Training Epoch: {i+1}/{self.cfg.stage1_num_epoch}') as pbar:
                    for action in self.trainDataloader:
                        action = action.to(torch.float32)
                        action = action.to(self.device)

                        with torch.cuda.amp.autocast():
                            loss, reconLoss, KLloss, mmd = self._run_single_step(action)
                            if torch.isnan(loss) or torch.isinf(loss):
                                cprint(f"Warning: loss is NaN or Inf! Loss={loss}", "red")  
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()

                        if logger_cnt == 9 and self.cfg.wandb_flag:
                            self._report(loss, reconLoss, KLloss, mmd)
                        
                        pbar.set_postfix({"Loss":f'{loss.item():.06f}', "KLD":f'{KLloss.item():.06f}', "MMD":f'{mmd.item():.06f}'})
                        pbar.update(1)
                        self.traing_dict["epoch"] = i
                        logger_cnt = (logger_cnt + 1) % 10

                if i % self.cfg.SA_step == 0 and self.optimizer.param_groups[0]['lr'] > self.cfg.min_lr:
                    self.scheduler.step()

                if i % self.cfg.eval_per_x_epoch == 0:
                    with tqdm(total=len(self.valDataloader), desc=f'Evaling  Epoch: {i+1}/{self.cfg.stage1_num_epoch}') as pbar:
                        self.model.eval()
                        meanloss = float("-inf")
                        total_loss = 0.0
                        cnt = 0

                        for action in self.valDataloader:
                            action = action.to(torch.float32)
                            action = action.to(self.device)
                            with torch.cuda.amp.autocast():
                                loss, reconLoss, KLloss, mmd = self._run_single_step(action)
                            pbar.set_postfix({"Loss":f'{loss.item():.06f}', "KLD":f'{KLloss.item():.06f}', "MMD":f'{mmd.item():.06f}'})
                            pbar.update(1)
                            total_loss += loss.item()
                            cnt += 1

                    meanloss = total_loss / cnt
                    if meanloss < self.traing_dict["BestLoss"]:
                        self.traing_dict["BestLoss"] = meanloss
                        self.BestModel = copy.deepcopy(self.model)
                print("self.cfg.save_every_x_epoch", self.cfg.save_every_x_epoch)
                if i > 0 and i % self.cfg.save_every_x_epoch == 0:
                    self.save()
            self.loaded_mu, self.loaded_std = self.compute_mu_and_std()
            self.save()
            if self.cfg.auto_stage2:
                self.traing_dict["stage"] = "Stage2"
                self.switch_stage("Stage2")
            else:
                np.savetxt(os.path.join(self.cfg.stage1_dataset_dir,'mu.txt'),
                        self.loaded_mu, fmt="%.6f")
                np.savetxt(os.path.join(self.cfg.stage1_dataset_dir,'std.txt'),
                            self.loaded_std, fmt="%.6f")
            
        if self.traing_dict["stage"] == "Stage2":
            self.LossFunc = Mldloss(True)
            if not self.cfg.enable_resume and self.traing_dict["epoch"] != 0:
                self.switch_stage("Stage2")
                self.init_model()

            for i in range(self.traing_dict["epoch"], self.cfg.stage2_num_epoch):
                self.model.train()
                logger_cnt = 0
                with tqdm(total=len(self.trainDataloader), desc=f'Training Epoch: {i+1}/{self.cfg.stage2_num_epoch}') as pbar:
                    for action in self.trainDataloader:
                        action = action.to(torch.float32)
                        action = action.to(self.device)

                        with torch.cuda.amp.autocast():
                            loss, reconLoss, KLloss, mmd = self._run_single_step(action)
                            if torch.isnan(loss) or torch.isinf(loss):
                                cprint(f"Warning: loss is NaN or Inf! Loss={loss}", "red")  
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()

                        if logger_cnt == 9 and self.cfg.wandb_flag:
                            self._report(loss, reconLoss, KLloss, mmd)
                        
                        pbar.set_postfix({"Loss":f'{loss.item():.06f}', "KLD":f'{KLloss.item():.06f}', "MMD":f'{mmd.item():.06f}'})
                        pbar.update(1)
                        self.traing_dict["epoch"] = i
                        logger_cnt = (logger_cnt + 1) % 10

                if i % self.cfg.SA_step == 0 and self.optimizer.param_groups[0]['lr'] > self.cfg.min_lr:
                    self.scheduler.step()

                if i % self.cfg.eval_per_x_epoch == 0:
                    with tqdm(total=len(self.valDataloader), desc=f'Evaling  Epoch: {i+1}/{self.cfg.stage2_num_epoch}') as pbar:
                        self.model.eval()
                        meanloss = float("-inf")
                        total_loss = 0.0
                        cnt = 0

                        for action in self.valDataloader:
                            action = action.to(torch.float32)
                            action = action.to(self.device)
                            with torch.cuda.amp.autocast():
                                loss, reconLoss, KLloss, mmd = self._run_single_step(action)
                            pbar.set_postfix({"Loss":f'{loss.item():.06f}', "KLD":f'{KLloss.item():.06f}', "MMD":f'{mmd.item():.06f}'})
                            pbar.update(1)
                            total_loss += loss.item()
                            cnt += 1

                    meanloss = total_loss / cnt
                    if meanloss < self.traing_dict["BestLoss"]:
                        self.traing_dict["BestLoss"] = meanloss
                        self.BestModel = copy.deepcopy(self.model)
                if i > 0 and i % self.cfg.save_every_x_epoch == 0:
                    self.save()
            self.save()
