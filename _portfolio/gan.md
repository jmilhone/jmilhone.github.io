---
title: "Training a Generative Adversarial Network (GAN)"
excerpt: "I try to train a GAN on the LSUN bedroom dataset.<br/><img src='/images/gan_final_image.png'>"
collection: portfolio
---

![GAN Training Progress](/images/bedroom_generated_images.gif)

I try to train a generative adverersarial network (GAN) with mixed success. My dataset is the LSUN bedroom dataset. I used less than 10% of the images because the dataset is massive and wanted to use the simple tensorflow dataset api function from_tensor_slices.

The architechture is based on DCGAN and consists of a generator and a discrimator for 64x64 pixel images. The generator starts with a 100 dimensional latent space. The first layer is a dense layer with 16384 nodes and is reshaped into a 4x4 image with 1024 channels. There are a series of upsample layers that increase the dimension by a factor of 2 and decrease the channels by a factor of 2. Each layer has leaky rectified linear units (LeakyReLU) for the activation followed by batch normalization. The final layer has a tanh activation layer. For the discriminator, I downsampled using padded covolutions with a stride of 2. Each layer has LeakyReLU activation functions and batch normalization as well. The final output layer is a single node with a sigmoid activation. 

The discriminator was trained with both real images and fake images from the generator. The discriminator's loss function was binary cross entropy for the real and fake images. The generator's loss function was binary cross entropy for the fake images.

I tried a few different things to help with training and got moderate success.

* Invert labels: real=0, fake=1
* Label smoothing: real=[0, 0.3], fake=[0.7,1.0]
* Noisy labels: 5% chance of a real image being labeled as fake for the discriminator training
* Adding a small amount of Gaussian noise to the real images
 
