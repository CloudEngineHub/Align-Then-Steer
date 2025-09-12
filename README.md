# Align-Then-stEer: Adapting the Vision-Language Action Models through Unified Latent Guidance

<p align="left">
  <a href="https://align-then-steer.github.io/">
    <img
      src="https://img.shields.io/badge/Align--Then--stEer-Website-87CEEB?logo=robotframework&logoColor=white"
      alt="ATE Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2509.02055">
    <img
      src="https://img.shields.io/badge/Align--Then--stEer-Paper-red?logo=arxiv&logoColor=red"
      alt="ATE Paper on arXiv"
    />
  </a>
</p>

We introduce **Align-Then-stEer (ATE)**, a novel, data-efficient, and plug-and-play adaptation framework that addresses the critical challenge of adapting Vision-Language-Action (VLA) models to downstream tasks. When the robot's embodiment or the task itself differs from the pre-training data, significant action distribution mismatches require extensive data and computational resources for effective fine-tuning. Our method aligns disparate action spaces by constructing a unified latent space and steers the generation process towards the target domain.

## ✨ Key Features

- 🔄 **Unified Latent Space Alignment**: Constructs a unified latent space through variational autoencoders, embedding adaptation actions into modes of the pre-training action latent distribution
- 🎯 **Guidance Mechanism**: Guides diffusion- or flow-based VLA generation processes during fine-tuning through a guidance mechanism
- 📊 **Data Efficiency**: Significantly boost the performance compared to direct fine-tuning, without demanding additional data
- 🔌 **Plug-and-Play**: Lightweight solution that is easy to integrate into any score-based VLAs
- 🌍 **Cross-Embodiment Adaptation**: Excellent performance in cross-embodiment and cross-task manipulation in both simulation and real-world settings

## 📈 Performance Improvements

- **Simulation Environment**: Improves average multi-task success rate by up to **9.8%** compared to direct fine-tuning of representative VLAs
- **Real-World**: Achieves a striking **32% success rate gain** in cross-embodiment settings

## 🔧 Technical Implementation

The ATE framework consists of two stages:

1. **Alignment Phase (Align)**: Uses a variational autoencoder constrained by reverse KL divergence to embed adaptation actions into modes of the pre-training action latent distribution
2. **Adaptation Phase (Steer)**: Pushes the model's output distribution towards the target domain through a classifier guidance

## 🎯 Applications

- Cross-embodiment robotic manipulation
- Cross-task adaptation
- Rapid deployment to new robotic platforms
- Efficient VLA adaptation when data collection is costly or labor-intensive

## 📅 Checklist

- [ ] Release code for Diffusion Policy (DP) with ATE
- [ ] Release code for RDT-1B with ATE on RoboTwin

## 🔥 Training VAE on Your Own Dataset
1. **Install**:
  Navigate to the Projects/ATE_vae directory and run：
  ```bash
  conda create -n ATE python==3.10
  conda activate ATE
  pip install -r requirements.txt
  ```

2. **Prepare your dataset**:
The Dataset should look like:
   ```bash
   |---- target_dir
        |---- hdf5
            |---- qpos (float)
        |---- other hdf5
        |---- ....
   ```
   Here, qpos has the shape (n, robot_dof), where n denotes the trajectory length, and robot_dof denotes the number of degrees of freedom (DoFs) of the robotic arm.
The pre-training data is exactly the same in form with the adaptation data.
3. **Prepare your yaml file**: we will next introduce the fields in the YAML file that typically need to be configured according to specific requirements. For the more detailed meanings of other fields, please refer to the comments in the YAML file.
- **pretraining_dataset_dir** and **adaptation_dataset_dir**: Specify the datasets required for training the VAE model in Step 1 (i.e., pre-training dataset) and Step 2 (i.e., adaptation dataset), respectively. The structure of the dataset directory you provide should follow the format described in the first point above.
- **s_length**: Defines the length of the Action chunk for the subsequent VLA model.
- **in_channels**: Represents the action dimension of the robot, usually equal to the sum of the degrees of freedom (DoF) of the robotic arm and the end-effector.
- **latent_dim**: Specifies the dimensionality of the latent space to which the VAE encodes the data.
- **isAuto**: When set to True, the program will automatically execute training for both stages.
- **resume**: Enables resuming training from a checkpoint in case of unexpected interruptions. To use this, set enable_resume to True in the YAML file and provide pretrained_model_dir.

After finishing the vae training, you may choose either the Best or the Last model to serve as the guidance model for RDT. The Best model always corresponds to the one with the lowest loss.

4. **Start Training**:
```
./scripts/train.sh 
```

## 🔥 Fine-Tuning RDT-ATE on Your Own Dataset
1. **Prepare your yaml file**
Before starting the training of RDT-VAE, you must first train a VAE model using the pretrained data of RDT along with the dataset of the downstream task. 
Click [here](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/main/README.md) to follow the RDT documentation, adjust the implementation of your dataset reader class, and configure the parameters accordingly.
In addition to the parameters that need to be adjusted for RDT training, you also need to add the following fields in config/base.yaml of the RDT project:
```yaml
model:
  
  ...

  vae:
    pretrained_path: /path/to/vae/checkpoint.pth
    mask: [0, 1, 2, 3, 4, 5, 10, 51, 52, 53, 54, 55, 60]
    lambda: 3.0
    clamp: 0.1
    input_channels: 12
    latent_dim: 512 
```

- **pretrained_path**: Directly points to the .pth file of the model weights you intend to use.
- **mask**: Specifies which dimensions to use within the 128-dimensional Unified Action Space of RDT.
- **lambda** parameters and **clamp** parameters: In most cases, these hyperparameters depend on your own dataset.
  - The **clamp** parameter specifically controls the volume to which the computed gradient term is clipped
- **input_channels** and **latent_dim**: Have the same meanings as described in the VAE training section above.
- **s_length**: In this stage, you do not need to explicitly configure the VAE’s s_length. However, you must ensure that s_length is consistent with the Action chunk length you expect for RDT.

2. **Start Training**:
```
./finetune.sh 
```