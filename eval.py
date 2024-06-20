from matplotlib import pyplot as plt
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import random
from PIL import Image
from PIL import ImageEnhance
from torchvision import transforms
import torchvision.transforms.functional as TF
from skimage.filters import gaussian
from skimage.filters import unsharp_mask
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torchvision.utils as vutils
from barbar import Bar

from IM2IM_model import Generator, Encoder
from preprocess import Preprocessing, CustomDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = './model_new.pth' 
checkpoint = torch.load(checkpoint_path)

G = Generator(4).to(device)
E = Encoder(4).to(device)

G.load_state_dict(checkpoint['Generator_state_dict'])
E.load_state_dict(checkpoint['Encoder_state_dict'])
G.eval()
E.eval()


dataset_path = "C:/Users/arush/Downloads/loading/inf_seen" #Update path to the folder containing the dataset
paired_data = Preprocessing.load_dataset(dataset_path)

custom_dataset = CustomDataset(paired_data)
batch_size = 1
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)

reconstruction_loss = nn.MSELoss()
for input_image, output_image in Bar(data_loader):

    input_image = input_image.to(device)
    output_image = output_image.to(device)

    # Normalize input and output images to range [0, 1]
    input_image = input_image.float()
    output_image = output_image.float()
    
    # Generate fake images
    encoder_output = E(input_image)
    z_fake = encoder_output[0]
    mu = encoder_output[1]
    log_sigma = encoder_output[2]
    
    x_generated = G(z_fake.view(z_fake.size(0), 4, 1, 1))
    rec_loss = reconstruction_loss(x_generated, output_image)
    print("reconstruction loss between target image and generated image:", rec_loss)

    Preprocessing.visualize(input_image, output_image, x_generated)
                

       
