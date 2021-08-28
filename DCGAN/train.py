import os
import argparse
import logging

import torch
import torch.nn as nn

from model import *
from dataset import *
from utils import *

def train():

    G = Generator(out_channel = nch).to(device)
    D = Discriminator(in_channel = nch).to(device)

    loss_fn = nn.BCELoss().to(device)

    G.apply(init_weights)
    D.apply(init_weights)

    optimizer_G = torch.optim.Adam(G.parameters(), lr = lr, betas = (beta1, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr = lr, betas = (beta1, 0.999))
