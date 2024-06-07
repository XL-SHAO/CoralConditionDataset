import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_channel):
        super(Discriminator, self).__init__()
        # Define your layers
       
        self.conv_1 = nn.Conv2d(input_channel, 256, kernel_size=4, stride=2, padding=1)
        self.leaky_relu_1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_2 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.leaky_relu_2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_3 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.leaky_relu_3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.conv_4 =nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1)
   

    def forward(self, x):
        x = self.leaky_relu_1(self.conv_1(x))
        x = self.leaky_relu_2(self.conv_2(x))
        x = self.leaky_relu_3(self.conv_3(x))
        x = self.conv_4(x)
        return x