# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
from utils import load_mnist_data
from models import LeNet5

import torch
import torch.optim as optim
import torch.nn as nn

import time

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == "__main__":

    device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    train_loader, test_loader = load_mnist_data(batch_size=64, root="../data")
    criterion = nn.CrossEntropyLoss()
    """
    # Instantiate the LeNet-5 model for GPU
    lenet5_gpu = LeNet5(device='cuda')

    # Define a loss function and optimizer

    optimizer = optim.Adam(lenet5_gpu.parameters(), lr=0.01)

    # Train and evaluate on GPU
    lenet5_gpu.to(device_gpu)
    start_time_gpu = time.time()
    lenet5_gpu.fit(train_loader, criterion, optimizer, epochs=5)
    test_accuracy_gpu = lenet5_gpu.evaluate(test_loader)
    end_time_gpu = time.time()
    time_gpu = end_time_gpu - start_time_gpu
    print(f'GPU Test Accuracy: {test_accuracy_gpu * 100:.2f}%, Training Time: {time_gpu:.2f} seconds')
 """

    device_cpu = torch.device('cpu')

    # Instantiate the LeNet-5 model for CPU
    lenet5_cpu = LeNet5(device='cpu')

    # Define a loss function and optimizer
    optimizer = optim.Adam(lenet5_cpu.parameters(), lr=0.01)

    # Train and evaluate on CPU
    start_time_cpu = time.time()
    lenet5_cpu.fit(train_loader, criterion, optimizer, epochs=5)
    test_accuracy_cpu = lenet5_cpu.evaluate(test_loader)
    end_time_cpu = time.time()
    time_cpu = end_time_cpu - start_time_cpu
    print(f'CPU Test Accuracy: {test_accuracy_cpu * 100:.2f}%, Training Time: {time_cpu:.2f} seconds')
   

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------