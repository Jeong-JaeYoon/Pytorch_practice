import os
import argparse
import logging

import torch
from torch._C import device
import torch.nn as nn

from model import *
from dataset import *
from utils import *

def train(dataloader):

    G = Generator(out_channel = nch).to(device)
    D = Discriminator(in_channel = nch).to(device)

    loss_fn = nn.BCELoss().to(device)

    G.apply(init_weights)
    D.apply(init_weights)

    optimizer_G = torch.optim.Adam(G.parameters(), lr = lr, betas = (beta1, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr = lr, betas = (beta1, 0.999))

    real_label = 1
    fake_label = 0

    for epoch in range(epochs):

        for i, data in enumerate(dataloader, 0):
            D.zero_grad()
            real_img = data['image'].to(device)
            input = torch.randn(real_img.shape[0], 100, 1, 1).to(device)
            