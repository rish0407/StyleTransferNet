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
import argparse 
from barbar import Bar

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Loading of the training dataset'''

class Preprocessing():
        
    @classmethod
    def load_image(cls, image_path):
        return Image.open(image_path).convert("RGB")

    @classmethod
    def load_dataset(cls, dataset_path): 
        #num_augmentations can be changed depending on task
        paired_data = []
        pair_folders = os.listdir(dataset_path)
        pair_folders.sort()

        for folder in pair_folders:
            folder_path = os.path.join(dataset_path, folder)
            
            if not os.path.isdir(folder_path):
                continue

            image_files = os.listdir(folder_path)
            image_files.sort()
            
            image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

            if len(image_files) == 2:
                output_image = cls.load_image(os.path.join(folder_path, image_files[0]))
                input_image = cls.load_image(os.path.join(folder_path, image_files[1]))

                input_image = input_image.resize(output_image.size, resample=Image.Resampling.BILINEAR)

                # Convert input and output images to numpy arrays
                input_image = np.array(input_image, dtype=np.double) / 255.0  # Normalize to range [0, 1]
                output_image = np.array(output_image, dtype=np.float64) / 255.0

                input_image = TF.to_pil_image(input_image)
                output_image = TF.to_pil_image(output_image)

                resize = transforms.Resize(size=(256, 256))
                input_image = resize(input_image)
                output_image = resize(output_image)
                
                input_image = TF.to_tensor(input_image)
                output_image = TF.to_tensor(output_image)

                paired_data.append((input_image, output_image))

        return paired_data

    @classmethod
    def visualize(cls, input_image, output_image, generated_image):

        fig, axes = plt.subplots(1, 3)
                    
        # Convert input image to numpy array for visualization
        input_image_numpy = input_image[0].permute(1, 2, 0).cpu().numpy()
        axes[0].imshow(input_image_numpy)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # Convert output image to numpy array for visualization
        output_image_numpy = output_image[0].permute(1, 2, 0).cpu().numpy()
        axes[1].imshow(output_image_numpy)
        axes[1].set_title('Output Image')
        axes[1].axis('off')

        generated_image = generated_image.clamp(0, 1)
        generated_image_numpy = generated_image[0].permute(1, 2, 0).detach().cpu().numpy()
        axes[2].imshow(generated_image_numpy)
        axes[2].set_title('Generated Image')
        axes[2].axis('off')

        # plt.imsave(output_images[i], dehazed_image_numpy)
        plt.show()


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_image, output_image = self.pairs[idx]
        return input_image, output_image