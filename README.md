# A custom VAE-GAN architecture called StyleTransferNet to perform image to image style transfer
The model includes components of vae and gans as well as their loss function to perform the image translation task of converting a cubist image to an impressionist image. The model is trained on training dataset consisting of 80 image pairs and validated on 40 image pairs. In order to create the image pairs, a base image was input into diffusion model to generate an impressionist and a cubist image. The image pairs are preprocessed as well.

Further details of preprocessing, model architecture and training have been provided in the Generative AI.pdf provided. 
