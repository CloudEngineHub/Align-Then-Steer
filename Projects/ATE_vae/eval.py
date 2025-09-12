# from models.mld_vae import EDMldVae, DDMldVae, KLLoss
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.info_vae import InfoVAE, Mldloss 
from utils.Dataset import RDTSeqDataset

import torch
from torch.utils.data import DataLoader

import os
from termcolor import cprint
from utils.functions import merge_yaml
from utils.functions import calculate_movement_fid
from omegaconf import OmegaConf
import numpy as np
import argparse
from tqdm import tqdm
import math


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--config_file', type=str, default="./configs/info.yaml", help='config file path')
    args = parser.parse_args()

    cfg = merge_yaml((OmegaConf.load("configs/base.yaml"), OmegaConf.load(args.config_file)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valDataset = RDTSeqDataset(os.path.join(cfg.stage1_dataset_dir,"val"), action_seq_length=cfg.s_length) if cfg.eval_stage == "Stage1" \
                else RDTSeqDataset(os.path.join(cfg.stage2_dataset_dir,"val"), action_seq_length=cfg.s_length)

    valDataloader = DataLoader(valDataset, batch_size=cfg.batch_size,
                            shuffle=True, num_workers=4,
                            pin_memory=True, drop_last=True)
    
    model = InfoVAE(latent_dim=[1, cfg.latent_dim], dropout=0.1)

    checkpoint = torch.load(os.path.join(cfg.pretrained_model_dir, f"{cfg.eval_target}.pth"))
    checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)

    cprint("Compiling model...", "green")
    torch.compile(model)
    cprint("Compiled...", "green")
    model.to(device)

    model.eval()
    if cfg.eval_stage == "Stage2":
        loaded_mu  = torch.tensor(np.loadtxt(os.path.join(cfg.pretrained_model, "mu.txt"), dtype=np.float32),
                                   dtype=torch.float32).unsqueeze(0).expand(np.batch_size, np.latent_dim).to(device)
        loaded_std  = torch.tensor(np.loadtxt(os.path.join(cfg.pretrained_model, "std.txt"), dtype=np.float32),
                                   dtype=torch.float32).unsqueeze(0).expand(np.batch_size, np.latent_dim).to(device)
    LossFunc = Mldloss(False) if cfg.eval_stage == "Stage2" else Mldloss(True)

    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_fid = 0
    total_mmd = 0
    cnt = 0

    real_motions = []
    generated_motions = []

    # for action in tqdm(valDataloader, total = 10000):  #=math.ceil(len(valDataset)/batch_size)):
    for action in tqdm(valDataloader, total=math.ceil(len(valDataset) / cfg.batch_size)):
        action = action.to(torch.float32)
        action = action.to(device)
        length_list = [cfg.s_length]*cfg.batch_size

        motion_z, dist_m, *_ = model.encode(action)

        temp = motion_z.permute(1, 0, 2).detach().view(motion_z.size(1), -1)
        real_motions.append(temp)

        feats_rst = model.decode(motion_z, length_list)
        feats_cpu = feats_rst.detach()

        gen_motion_z, *_ = model.encode(feats_rst)
        
        temp = gen_motion_z.permute(1, 0, 2).detach().view(gen_motion_z.size(1), -1)
        generated_motions.append(temp)

        mu_ref = torch.zeros_like(dist_m.loc)
        scale_ref = torch.ones_like(dist_m.scale)
        
        dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        
        feats = {
                "m_ref": action,
                "m_rst": feats_rst,
                "dist_m": dist_m,
                "dist_ref": dist_ref,
                "z": motion_z
            }
        loss, reconLoss, KLloss, mmd = LossFunc(feats)

        total_loss += loss.item()
        total_recon_loss += reconLoss.item()
        total_kl_loss += KLloss.item()
        total_mmd += mmd.item()
        cnt+=1
        if cnt >= cfg.eval_step:
            break

    
    diversity_output = []
    for action in valDataloader:
        ction = action.to(torch.float32)
        action = action.to(device)
        length_list = [cfg.s_length] * cfg.batch_size

        for i in range(1000):
            motion_z, dist_m, *_ = model.encode(action)
            feats_rst = model.decode(motion_z, length_list)

            diversity_output.append(feats_rst.detach())

        break
    diversity = torch.var(torch.stack(diversity_output), dim=0)
    avd_diversity =diversity.mean().item()


    real_motions = torch.stack(real_motions, dim=0)
    generated_motions = torch.stack(generated_motions, dim=0)
    real_motions = real_motions.view(-1, real_motions.size(2)).to("cpu").numpy()
    generated_motions = generated_motions.view(-1, generated_motions.size(2)).to("cpu").numpy()
    FID = calculate_movement_fid(real_motions, generated_motions)

    cprint("LOSS: \t\t{:.6f}".format(total_loss/cnt), "red")
    cprint("RECON_LOSS: \t{:.6f}".format(total_recon_loss/cnt), "yellow")
    cprint("MMD: \t\t{:.6f}".format(total_mmd/cnt), "blue")
    cprint("KL_LOSS: \t{:.6f}".format(total_kl_loss/cnt), "green")
    cprint("MultiModality: \t{:.6f}".format(avd_diversity), "cyan")
    cprint("FID: \t\t{:.6f}".format(FID), "magenta")
    
