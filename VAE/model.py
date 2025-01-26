import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class VAE(nn.Module):
  def __init__(self, input_channels, hidden_dim, latent_dim):
    super(VAE, self).__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
    )
    self.fc1 = nn.Linear(512*16*16, latent_dim) # 평균
    self.fc2 = nn.Linear(512*16*16, latent_dim) # 분산
    self.fc3 = nn.Linear(latent_dim, 512*16*16) # 분산
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid()
    )

  def encode(self, x):
    h = self.encoder(x)
    h = h.view(h.size(0), -1)
    mu, log_var = self.fc1(h), self.fc2(h)
    return mu, log_var

  def reparmeterization(self, mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    return mu + eps*std

  def decode(self, z):
    h = self.fc3(z)
    h = h.view(h.size(0), 512, 16, 16)
    return self.decoder(h)

  def forward(self, x):
    mu, log_var = self.encode(x)
    z = self.reparmeterization(mu, log_var)
    return self.decode(z), mu, log_var