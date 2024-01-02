# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
from utils import load_mnist_data
from models import VariationalAutoencoder

import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the data
    train_loader, test_loader = load_mnist_data(batch_size=64, root="../data")

    # Instantiate the LeNet-5 model
    model = VariationalAutoencoder()
    model.fit(train_loader, losses = True)


    # Generate images from random noise
    with torch.no_grad():
        num_samples = 16
        latent_dim = 64  # Assuming latent space size is 64
        random_latent_samples = torch.randn(num_samples, latent_dim)
        generated_samples = model.decoder(random_latent_samples)
    
    # Reshape and display the generated samples
    generated_samples = generated_samples.view(-1, 28, 28)
    for i in range(generated_samples.size(0)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_samples[i].cpu().numpy(), cmap='gray')
        plt.axis('off')
    
    plt.show()
        