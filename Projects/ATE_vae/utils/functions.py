import os
import numpy as np
from scipy.linalg import sqrtm
from omegaconf import OmegaConf

def merge_yaml(yaml_dicts):
    base_yaml = yaml_dicts[0]
    for yaml in yaml_dicts:
        base_yaml = OmegaConf.merge(base_yaml, yaml)
    return base_yaml

def find_target(target_path):
    target_list = []
    sub_names = os.listdir(target_path)
    for sub_name in sub_names:
        sub_path = os.path.join(target_path, sub_name)
        if os.path.isdir(sub_path):
            target_list += find_target(sub_path)
        else:
            if sub_path.endswith(".hdf5"):
                target_list.append(sub_path)
    return target_list


def calculate_movement_fid(real_motions, generated_motions, eps=1e-6):
    """
    real_motions: shape [N1, 64, latent]
    generated_motions: shape [N2, 64, latent]
    """
    real_features = real_motions.reshape(real_motions.shape[0], -1) # [N1, 1*32]
    generated_features = generated_motions.reshape(generated_motions.shape[0], -1) # [N2, 1*32]

    mu_real = np.mean(real_features, axis=0) # [1*latent]
    mu_gen = np.mean(generated_features, axis=0) # [1*latent]

    mu_real = np.atleast_1d(mu_real) # [1*latent]
    mu_gen = np.atleast_1d(mu_gen) # [1*latent]

    sigma_real = np.cov(real_features, rowvar=False) # [1*32, 1*32]
    sigma_gen = np.cov(generated_features, rowvar=False) # [1*32, 1*32]
    
    sigma_real = np.atleast_2d(sigma_real) # [1*32, 1*32]
    sigma_gen = np.atleast_2d(sigma_gen) # [1*32, 1*32]

    diff = mu_real - mu_gen

    covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; "
               "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = sqrtm((sigma_real + offset).dot(sigma_gen + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * tr_covmean

    return fid