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

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def load_and_augment_dataset(dataset_path, num_augmentations=3): #num_augmentations can be changed depending on task
    augmented_pairs = []

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
            output_image = load_image(os.path.join(folder_path, image_files[0]))
            input_image = load_image(os.path.join(folder_path, image_files[1]))

            input_image = input_image.resize(output_image.size, resample=Image.Resampling.BILINEAR)

            # Convert input and output images to numpy arrays
            input_image = np.array(input_image, dtype=np.double) / 255.0  # Normalize to range [0, 1]
            output_image = np.array(output_image, dtype=np.float64) / 255.0

            for _ in range(num_augmentations):
                input_image_augmented, output_image_augmented = transform_images(input_image, output_image)

                augmented_pairs.append((input_image_augmented, output_image_augmented))

    return augmented_pairs

''' Augmentation of the dataset'''

def transform_images(input_image, output_image):

    # Convert numpy arrays to PIL images
    input_image = TF.to_pil_image(input_image)
    output_image = TF.to_pil_image(output_image)

    resize = transforms.Resize(size=(256, 256))
    input_image = resize(input_image)
    output_image = resize(output_image)

    #Randomly crops the pair of images
    i, j, h, w = transforms.RandomCrop.get_params(input_image, output_size=(256, 256))
    input_image = TF.crop(input_image, i, j, h, w)
    output_image = TF.crop(output_image, i, j, h, w)

    #Randomly horizontally flips the pair of images
    if random.random() > 0.5:
        input_image = TF.hflip(input_image)
        output_image = TF.hflip(output_image)

    #Randomly vertically flips the pair of images
    if random.random() > 0.5:
        input_image = TF.vflip(input_image)
        output_image = TF.vflip(output_image)

    #Randomly tils the pair of images
    affine_param = transforms.RandomAffine.get_params(
        degrees = [-180, 180], translate = [0.3,0.3],  
        img_size = [520, 520], scale_ranges = [1, 1.3], 
        shears = [2,2])
    input_image = TF.affine(input_image, 
                      affine_param[0], affine_param[1],
                      affine_param[2], affine_param[3])
    output_image = TF.affine(output_image, 
                     affine_param[0], affine_param[1],
                     affine_param[2], affine_param[3])

    #Randomly changes the brightness of the pair of images
    brightness_factor = random.uniform(0.8, 1.2)
    input_image = ImageEnhance.Brightness(input_image).enhance(brightness_factor)
    output_image = ImageEnhance.Brightness(output_image).enhance(brightness_factor)

    #Randomly changes the contrast of the pair of images
    contrast_factor = random.uniform(0.8, 1.2)
    input_image = ImageEnhance.Contrast(input_image).enhance(contrast_factor)
    output_image = ImageEnhance.Contrast(output_image).enhance(contrast_factor)

    #Randomly changes the saturation of the pair of images
    saturation_factor = random.uniform(0.8, 1.2)
    input_image = ImageEnhance.Color(input_image).enhance(saturation_factor)
    output_image = ImageEnhance.Color(output_image).enhance(saturation_factor)

    #Randomly changes the hue of the pair of images
    hue_factor = random.uniform(0.8, 1.2)
    input_image = ImageEnhance.Sharpness(input_image).enhance(hue_factor)
    output_image = ImageEnhance.Sharpness(output_image).enhance(hue_factor)

    input_image = np.array(input_image)
    output_image = np.array(output_image)

    #Randomly adds gaussian blur to the pair of images
    if random.random() < 0.25:
        sigma_param = random.uniform(0.1, 0.5)
        input_image = gaussian(input_image, sigma=sigma_param)
        output_image = gaussian(output_image, sigma=sigma_param)
    
    #Randomly adds unsharp mask to the pair of images
    if random.random() < 0.25:
        radius_param = random.uniform(0, 2)
        amount_param = random.uniform(0.2, 1)
        input_image = unsharp_mask(input_image, radius=radius_param, amount=amount_param)
        output_image = unsharp_mask(output_image, radius=radius_param, amount=amount_param)
    
    input_image = TF.to_tensor(input_image)
    output_image = TF.to_tensor(output_image)

    return input_image, output_image

dataset_path = "C:/Users/arush/Downloads/loading/train" #Update path to the folder containing the dataset
augmented_pairs = load_and_augment_dataset(dataset_path)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_image, output_image = self.pairs[idx]
        return input_image, output_image

custom_dataset = CustomDataset(augmented_pairs)
batch_size = 40
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

'''Initializing weights to be updated'''

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)


'''Discriminator Class'''

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.zero_pad1 = nn.ZeroPad2d(1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.zero_pad2 = nn.ZeroPad2d(1)
        self.last_conv = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0)
        self.drop = nn.Dropout2d(p = 0.4)

    def forward(self, inp, tar):

        print("Entering Discriminator")
        x_1 = torch.cat([inp, tar], dim=1)
        x_2 = self.leaky_relu(self.bn1(self.conv1(x_1)))
        x_3 = self.drop(x_2)
        x_4 = self.leaky_relu(self.bn2(self.conv2(x_3)))
        x_5 = self.drop(x_4)
        x_6 = self.leaky_relu((self.conv3(x_5)))
        x_7 = self.drop(x_6)
        x_8 = self.zero_pad1(x_7)
        x_9 = (self.last_conv(x_8))
        x_10 = self.drop(x_9)
        print("Leaving Discriminator")
        return torch.sigmoid(x_10)

'''Generator Class'''

class Generator(nn.Module):
    def __init__(self, z_dim=50):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.output_bias = nn.Parameter(torch.zeros(3, 32, 32), requires_grad=True)
        self.deconv1 = nn.ConvTranspose2d(z_dim, 32, 4, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 4, stride=4, padding = 0, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.deconv3 = nn.ConvTranspose2d(16, 8, 4, stride=4,padding = 0, bias=False)
        self.bn3 = nn.BatchNorm2d(8)
        self.deconv4 = nn.ConvTranspose2d(8, 3, 4, stride=4, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(3)

    def forward(self, z):
        print("Entering generator")
        z_1 = F.leaky_relu(self.bn1(self.deconv1(z)), negative_slope=0.1)
        z_2 = F.leaky_relu(self.bn2(self.deconv2(z_1)), negative_slope=0.1)
        z_3 = F.leaky_relu(self.bn3(self.deconv3(z_2)), negative_slope=0.1)
        z_4 = F.leaky_relu(self.bn4(self.deconv4(z_3)), negative_slope=0.1)
        print("Leaving Generator")
        return z_4

'''Encoder Class'''

class Encoder(nn.Module):
    def __init__(self, z_dim=50):
        super(Encoder, self).__init__()
        self.z_dim = z_dim

        #Block 1
        self.conv1_1 = nn.Conv2d(3, 32, 5, stride=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 64, 4, stride=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.maxPool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #Block 2
        self.conv2_1 = nn.Conv2d(64, 128, 4, stride=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 256, 4, stride=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(256)

        self.maxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #Block 3
        self.conv3_1 = nn.Conv2d(256, 512, 4, stride=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(512)
        self.conv3_2 = nn.Conv2d(512, 512, 1, stride=1, padding='same', bias=True, padding_mode='reflect')
        self.bn3_2 = nn.BatchNorm2d(512)

        self.maxPool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        #Dropout added because encoder becoming computationally expensive
        self.drop = nn.Dropout2d(p = 0.4)
        #Fully Connected layer.
        self.fc = nn.Linear(240, self.z_dim*2)


    def forward(self, x):

        print("Entering encoder")
        x1_1 = F.leaky_relu(self.bn1_1(self.conv1_1(x)), negative_slope=0.1)
        x1_2 = F.leaky_relu(self.bn1_2(self.conv1_2(x1_1)), negative_slope=0.1)
        x1_3 = self.drop(x1_2)
        x1 = self.maxPool_1(x1_3)
        print("Block 1 complete")

        x2_1 = F.leaky_relu(self.bn2_1(self.conv2_1(x1)), negative_slope=0.1)
        x2_2 = F.leaky_relu(self.bn2_2(self.conv2_2(x2_1)), negative_slope=0.1)
        x2_3 = self.drop(x2_2)
        x2 = self.maxPool_2(x2_3)
        print("Block 2 complete")

        x3_1 = F.leaky_relu(self.bn3_1(self.conv3_1(x2)), negative_slope=0.1)
        x3_2 = F.leaky_relu(self.bn3_2(self.conv3_2(x3_1)), negative_slope=0.1)
        x3_3 = self.drop(x3_2)
        x3 = self.maxPool_3(x3_3)
        print("Block 3 complete")
        
        '''Flattening and slicing x to be passed into linear fc layer'''
        x_flat = torch.flatten(x3, start_dim=1)
        x_slice = x_flat[:, :240]
    
        x_fc = self.fc(x_slice)
        z = self.reparameterize(x_fc)
        return z

    '''Applying the reparameterization trick to sample latent vector'''
    def reparameterize(self, z):
        mu, log_sigma = z[:, :self.z_dim], z[:, self.z_dim:]
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        z_out = mu + eps * std
        return z_out, mu, log_sigma

torch.autograd.set_detect_anomaly(True)

'''Trainer Class for training the model'''
class Trainer:
    def __init__(self, args, data, device, checkpoint_dir):
        self.args = args
        self.train_loader = data
        self.device = device
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):
        """Training the hybrid model"""
        self.G = Generator(self.args.latent_dim).to(self.device)
        self.E = Encoder(self.args.latent_dim).to(self.device)
        self.D = Discriminator().to(self.device)
        
        self.G.apply(weights_init_normal)
        self.E.apply(weights_init_normal)
        self.D.apply(weights_init_normal)

        # checkpoint_path = './checkpoint_try2_epoch_1' 
        # checkpoint = torch.load(checkpoint_path)

        # self.G.load_state_dict(checkpoint['Generator_state_dict'])
        # self.E.load_state_dict(checkpoint['Encoder_state_dict'])
        # self.D.load_state_dict(checkpoint['Discriminator_state_dict'])

        optimizer_ge = optim.Adam(list(self.G.parameters()) +
                                      list(self.E.parameters()), lr=self.args.lr_adam, weight_decay=0.001)
        optimizer_d = optim.Adam(self.D.parameters(), lr=self.args.lr_adam, weight_decay=0.001)

        fixed_z = Variable(torch.randn((16, self.args.latent_dim, 1, 1)),
                           requires_grad=False).to(self.device)
        
        
        criterion = nn.BCELoss()
        reconstruction_loss = nn.MSELoss()

        for epoch in range(self.args.num_epochs+1):
            ge_losses = 0
            d_losses = 0

            for input_image, output_image in Bar(self.train_loader):

                input_image = input_image.to(self.device)
                output_image = output_image.to(self.device)

                # Normalize input and output images to range [0, 1]
                input_image = input_image.float()
                output_image = output_image.float()
                #print("Size of input_image before passing to the encoder:", input_image.size())

                optimizer_ge.zero_grad()
                optimizer_d.zero_grad()

                self.D.zero_grad()
                # Train discriminator with real images
                out_true = self.D(input_image, output_image)
                y_true = torch.ones_like(out_true)
                d_real_loss = criterion(out_true, y_true)

                # Generate fake images
                encoder_out = self.E(input_image)
                z_fake = encoder_out[0]
                mu = encoder_out[1]
                log_sigma = encoder_out[2]

                #Calculating KL Divergence between latent and prior distributipn
                kl_loss = -0.5 * torch.sum(1+ log_sigma - mu.pow(2) - torch.exp(log_sigma), dim = -1)
                kl_loss = torch.mean(kl_loss)

                x_generated = self.G( z_fake.view(z_fake.size(0), 50, 1, 1))
                out_fake = self.D(input_image, x_generated)
                y_fake = torch.ones_like(out_fake)
                d_fake_loss = criterion(out_fake, y_fake)
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward(retain_graph=True)
                optimizer_d.step()
                
                # Update generator
                self.G.zero_grad()
                #Train generator to fool discriminator
                out_true = self.D(input_image, output_image)
                out_fake = self.D(input_image, x_generated)
                dll_rec_loss = reconstruction_loss(out_fake, out_true)
                g_adv = criterion(out_fake, y_true)
                #Kl div getting tensor loss, need scalar loss.
                ge_loss = g_adv + dll_rec_loss + 0.5*kl_loss
                ge_loss.backward()
                optimizer_ge.step()
                
                #Computing gradients and backpropagate           
                d_losses += d_loss.item()
                ge_losses += ge_loss.item()

            print("Training... Epoch: {}, Discriminator Loss: {:.3f}, Generator Loss: {:.3f}".format(
                epoch, d_losses/len(self.train_loader), ge_losses/len(self.train_loader)
            ))

            # with torch.no_grad():
            #     encoder_output = self.E(input_image)
            #     z_out = encoder_output[0]
            #     x_ge = self.G( z_out.view(z_out.size(0), 50, 1, 1))
            #     self.visualize(input_image, output_image, x_ge)

            if self.args.save_checkpoint:
                self.save_checkpoint(epoch)
        
        self.save_checkpoint('final')
    
    def save_checkpoint(self, epoch):
        if epoch == 'final':
            checkpoint_path = os.path.join(self.checkpoint_dir, 'final_model_part2.pth')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_try2_epoch_{epoch}.pth')

        torch.save({
            'epoch': epoch,
            'Generator_state_dict': self.G.state_dict(),
            'Encoder_state_dict': self.E.state_dict(),
            'Discriminator_state_dict': self.D.state_dict(),    
        }, checkpoint_path)


    '''Visualizing the input, target and generated images'''

    def visualize(self, input_image, output_image, generated_image):

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

        # Clip and normalize generated image pixel values to range [0, 1]
        # generated_image_clipped = torch.clamp(generated_image, min=0, max=1)
        generated_image = generated_image.clamp(0, 1)
        generated_image_numpy = generated_image[0].permute(1, 2, 0).detach().cpu().numpy()
        # generated_image_numpy = generated_image[0].permute(1, 2, 0).detach().cpu().numpy()
        axes[2].imshow(generated_image_numpy)
        axes[2].set_title('Generated Image')
        axes[2].axis('off')

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="number of epochs")
    parser.add_argument('--lr_adam', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_rmsprop', type=float, default=1e-4,
                        help='learning rate RMSprop if WGAN is True.')
    parser.add_argument("--batch_size", type=int, default=40, 
                        help="Batch size")
    parser.add_argument('--latent_dim', type=int, default=50,
                        help='Dimension of the latent variable z')
    parser.add_argument('--save_checkpoint', type=bool, default=True,
                        help='If checkpoint to be saved')
    #parsing arguments.
    args = parser.parse_args() 

    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = "C:/Users/arush/Downloads/loading/train" #Update path to folder containing the dataset
    augmented_pairs = load_and_augment_dataset(dataset_path)
    custom_dataset = CustomDataset(augmented_pairs)
    batch_size = 40
    data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    checkpoint_dir = '.'

    model = Trainer(args, data_loader, device, checkpoint_dir)
    model.train()

