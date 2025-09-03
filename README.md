# Align-Then-stEer: Adapting the Vision-Language Action Models through Unified Latent Guidance

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