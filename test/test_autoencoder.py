# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
from utils import load_mnist_data
from models import LinearAutoencoder

import torch
import torch.optim as optim
import torch.nn as nn

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the data
    train_loader, test_loader = load_mnist_data(batch_size=64, root="../data")

    # Instantiate the LeNet-5 model
    model = LinearAutoencoder()
    model.fit(train_loader, losses = True)
    model.plot_loss_curve()
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



