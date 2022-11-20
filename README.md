# Latent-DDPMs
In this Repository, we test the performance of different backbones on Latent Diffusion task for Image generation.

# Tested Backbones

1. UNet architecture proposed in the original Paper.
2. Wide-ResNet Based Backbone Network
3. EfficientNetV3 based Backbone Network
4. VisionTransformer Based Backnoe Network

# Implementation Details
 1. Image size : We train on 256x256 sized images, but we use a pretrained network to bring the size of the embeddings down to 1x1x512. The diffusion is carried out in this Latent space.
 2. Schedule : We use a linear noise decay schedule
 3. Dataset : We train the network on :
    a. MNIST dataset.
    b. Subset of the WikiArt dataset.

# TODO :
1. Cite the original Paper
2. Cite datasets
3. Cite reference code implementations
