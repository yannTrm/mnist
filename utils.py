# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def download_mnist(root='./data'):
    # Define the transformation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Download the MNIST dataset
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

def load_mnist_data(root='./data', batch_size=64):
    # Define the transformation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root=root, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=False, transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_image(data_loader, num_image=0):
    # obtain one batch of training images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    images = images.numpy()
    
    # get one image from the batch
    img = np.squeeze(images[num_image])
    return img

def plot_image(image):
    fig = plt.figure(figsize=(5, 5)) 
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    plt.show()
    

# Afficher les résultats
def display_results(autoencoder, dataloader, num_images=5):
    autoencoder.eval()
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= num_images:
                break
            
            images, _ = data
            images = images.view(images.size(0), -1)
            latent, decoded_images = autoencoder(images)

            # Afficher les images originales
            plt.figure(figsize=(12, 6))
            plt.subplot(2, num_images, i + 1)
            plt.imshow(images.view(-1, 28, 28).numpy()[0], cmap='gray')
            plt.title('Original')
            plt.axis('off')

            # Afficher les images reconstruites
            plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(decoded_images.view(-1, 28, 28).numpy()[0], cmap='gray')
            plt.title('Reconstructed')
            plt.axis('off')

    plt.show()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------