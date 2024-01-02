# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class LeNet5(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(LeNet5, self).__init__()
        self.device = torch.device(device)
        self.__architecture__()
        
    def __architecture__(self):
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)  
        
    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x
    
    def fit(self, train_loader, criterion, optimizer, epochs):
        for epoch in range(epochs):
            running_loss = 0.0
            running_accuracy = 0.0

            for batch_idx, (data, labels) in enumerate(train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                accuracy = correct / labels.size(0)
                running_accuracy += accuracy

                running_loss += loss.item()
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}, Accuracy: {accuracy * 100:.2f}%')

            avg_loss = running_loss / len(train_loader)
            avg_accuracy = running_accuracy / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}, Average Accuracy: {avg_accuracy * 100:.2f}%')

    
    def evaluate(self, test_loader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                outputs = self(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class LinearAutoencoder(nn.Module):
    def __init__(self):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Tanh()
            )
        self.losses = []
        
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc, dec

    def fit(self, dataloader, num_epochs=10, learning_rate=2e-3, losses = False):
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)


        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for data in dataloader:
                img, labels = data
                img = img.view(img.size(0), -1)

                latent, output = self(img)
                
                loss = criterion(output, img)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
            if losses :    
                average_loss = epoch_loss / len(dataloader)
                self.losses.append(average_loss)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.data.item()}')
            
    def plot_loss_curve(self):
        plt.plot(range(1, len(self.losses) + 1), self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss Curve')
        plt.show()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(128, 64)
        self.fc_logvar = nn.Linear(128, 64)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Tanh()
        )
        
        self.losses = []
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc = self.encoder(x)
        
        mu = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)
        
        z = self.reparameterize(mu, logvar)
        
        dec = self.decoder(z)
        return dec, mu, logvar

    def fit(self, dataloader, num_epochs=10, learning_rate=2e-3, losses=False):
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for data in dataloader:
                img, labels = data
                img = img.view(img.size(0), -1)

                output, mu, logvar = self(img)
                
                # Reconstruction loss
                recon_loss = criterion(output, img)
                
                # KL divergence
                kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                loss = recon_loss + kl_divergence

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            if losses:
                average_loss = epoch_loss / len(dataloader)
                self.losses.append(average_loss)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Recon Loss: {recon_loss.data.item()}, KL Divergence: {kl_divergence.data.item()}')


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

        
