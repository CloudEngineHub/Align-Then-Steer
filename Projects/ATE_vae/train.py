from omegaconf import OmegaConf
from utils.trainer import trainer
from termcolor import cprint
import os
import argparse
from utils.functions import merge_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--config_file', type=str, default="./configs/info.yaml", help='config file path')
    args = parser.parse_args()

    cfg = merge_yaml((OmegaConf.load("configs/base.yaml"), OmegaConf.load(args.config_file)))

    cprint(OmegaConf.to_yaml(cfg), "light_green")
    model_trainer = trainer(cfg)
    
    try:
        model_trainer.fit()
    except:
        cprint(f'\nInterrupt Saving', on_color="on_red")
        model_trainer.save()
        os._exit(1)
