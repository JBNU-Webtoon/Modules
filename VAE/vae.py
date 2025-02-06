import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.patches as patches
from VAE.voc import VOCDataset
from VAE.loss import Loss

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class VAE(nn.Module):
  def __init__(self, input_channels, hidden_dim, latent_dim, feature_size=8, num_bboxes=2, num_classes=2):
    super(VAE, self).__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(input_channels, 16, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 45, kernel_size=1)
    ).to('cuda:1')
    self.fc1 = nn.Linear(45*16*16, latent_dim).to('cuda:1') # 평균
    self.fc2 = nn.Linear(45*16*16, latent_dim).to('cuda:1') # 분산
    self.fc3 = nn.Linear(latent_dim, 256*16*16).to('cuda:1') # 분산
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, input_channels, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid()
    ).to('cuda:1')
    self.detection_head = nn.Sequential(
        nn.Linear(45 * 16* 16 , 16* 16 * 256), # 차원 맞춰주기용?
        nn.Linear(16 * 16 * 256, 4096), # 여기를 latent dim으로 바꾸느냐 아니면 아래에서 이미지 encoder를 따로 돌리느냐의 차이
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(0.5, inplace=False),
        nn.Linear(4096, feature_size*feature_size*(5*num_bboxes + num_classes)),  # Class scores + bounding box (x, y, w, h)
        nn.Sigmoid()
    ).to('cuda:2')

  def encode(self, x):
    x = x.to('cuda:1')
    h = self.encoder(x)
    h = h.view(h.size(0), -1)
    mu, log_var = self.fc1(h), self.fc2(h)
    return mu, log_var

  def reparmeterization(self, mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std).to('cuda:1')
    return mu + eps*std

  def decode(self, z):
    z = z.to('cuda:1')
    h = self.fc3(z)
    h = h.view(h.size(0), 256, 16, 16)
    return self.decoder(h)

  def forward(self, x):
    # VAE Encoding
    x = x.to('cuda:1')
    mu, log_var = self.encode(x)
    z = self.reparmeterization(mu, log_var)
    # VAE Decoding
    recon = self.decode(z)
    h = self.encoder(x)
    h_flat = h.view(h.size(0), -1)
    # Detection
    h_flat = h_flat.to('cuda:2')
    detection_output = self.detection_head(h_flat)
    detection_output = detection_output.view(-1, 8, 8, 12)
    return h_flat, recon, mu, log_var, detection_output

def loss_function_MSE(recon_x, y, mu, log_var):
  MSE = nn.functional.mse_loss(recon_x, y, reduction='sum') # log scale 하기 
   # Apply log(1 + MSE)
  log_MSE = torch.log1p(MSE)  # log(1 + MSE)
  KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
  return log_MSE + KLD

