import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent = 100, out_channel = 3):
        super(Generator, self).__init__()
        self.convT1 = nn.ConvTranspose2d(in_channels = latent, out_channels = 512, kernel_size = 4, stride = 1, padding = 0, bias = False)
        self.norm1 = nn.BatchNorm2d(num_features = 512)
        
        self.convT2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.norm2 = nn.BatchNorm2d(num_features = 256)

        self.convT3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.norm3 = nn.BatchNorm2d(num_features = 128)

        self.convT4 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.norm4 = nn.BatchNorm2d(num_features = 64)

        self.convT5 = nn.ConvTranspose2d(in_channels = 64, out_channels = out_channel, kernel_size = 4, stride = 2, padding = 1, bias = False)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.norm1(self.convT1(x))
        x = self.relu(x)

        x = self.norm2(self.convT2(x))
        x = self.relu(x)

        x = self.norm3(self.convT3(x))
        x = self.relu(x)

        x = self.norm4(self.convT4(x))
        x = self.relu(x)

        x = self.convT5(x)
        x = self.tanh(x)

        return x 

class Discriminator(nn.Module):
    def __init__(self, in_channel = 3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channel, out_channels = 64, kernel_size = 4, stride = 2, padding = 1, bias = False)
                
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.norm2 = nn.BatchNorm2d(num_features = 128)

        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.norm3 = nn.BatchNorm2d(num_features = 256)

        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.norm4 = nn.BatchNorm2d(num_features = 512)

        self.conv5 = nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4, stride = 1, padding = 0, bias = False)
        
        self.leakyrelu = nn.LeakyReLU(0.2, inplace = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu(x)

        x = self.norm2(self.conv2(x))
        x = self.leakyrelu(x)

        x = self.norm3(self.conv3(x))
        x = self.leakyrelu(x)

        x = self.norm4(self.conv4(x))
        x = self.leakyrelu(x)

        x = self.conv5(x)
        x = self.sigmoid(x)

        return x