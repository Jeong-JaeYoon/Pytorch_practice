import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent = 100, ):
        super(Generator, self).__init__()
        self.convT1 = nn.ConvTranspose2d(in_channels = latent, out_channels =, kernel_size = 4, stride =, padding = 0, bias = False)
        self.norm1 = nn.BatchNorm2d()
        
        self.convT2 = nn.ConvTranspose2d()
        self.norm2 = nn.BatchNorm2d()

        self.convT3 = nn.ConvTranspose2d()
        self.norm3 = nn.BatchNorm2d()

        self.convT4 = nn.ConvTranspose2d()
        self.norm4 = nn.BatchNorm2d()

        self.convT5 = nn.ConvTranspose2d()

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
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d()
                
        self.conv2 = nn.Conv2d()
        self.norm2 = nn.BatchNorm2d()

        self.conv3 = nn.Conv2d()
        self.norm3 = nn.BatchNorm2d()

        self.conv4 = nn.Conv2d()
        self.norm4 = nn.BatchNorm2d()

        self.conv5 = nn.Conv2d()
        
        self.leakyrelu = nn.LeakyReLU(0.2, )
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