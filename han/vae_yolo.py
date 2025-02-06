import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from util_layers import Flatten


class VAE_YOLO(nn.Module):
    def __init__(self, input_channels, latent_dim, grid_size, bbox_atts, num_classes):
        super(VAE_YOLO, self).__init__()

        self.grid_size = grid_size
        self.bbox_atts = bbox_atts
        self.num_classes = num_classes

        # **Encoder**
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(192, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.fc_mu = nn.Linear(1024 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 7 * 7, latent_dim)
        self.fc_decoder_input = nn.Linear(latent_dim, 1024 * 7 * 7)

        # **Decoder**
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Normalizing to [0,1]
        )

        # **YOLO Head**
        self.yolo_head = nn.Sequential(
            Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(4096, grid_size * grid_size * (5 * bbox_atts + num_classes)),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decoder_input(z)
        h = h.view(h.size(0), 1024, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        # YOLO Output
        z_f = self.fc_decoder_input(z).view(-1, 1024, 7, 7)
        yolo_output = self.yolo_head(z_f)
        yolo_output = yolo_output.view(-1, self.grid_size, self.grid_size, 5 * self.bbox_atts + self.num_classes)
        return recon_x, yolo_output, mu, logvar

    # 나중에 쓸거 같아서 추가
    # def _make_conv_layers(self, bn):
    #     net = nn.Sequential(
    #         nn.Conv2d(1024, 1024, 3, padding=1),
    #         nn.LeakyReLU(0.1, inplace=True),
    #         nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
    #         nn.LeakyReLU(0.1),
    #
    #         nn.Conv2d(1024, 1024, 3, padding=1),
    #         nn.LeakyReLU(0.1, inplace=True),
    #         nn.Conv2d(1024, 1024, 3, padding=1),
    #         nn.LeakyReLU(0.1, inplace=True)
    #     )

    # def _make_fc_layers(self):
    #     S, B, C = self.feature_size, self.num_bboxes, self.num_classes
    #
    #     net = nn.Sequential(
    #         Flatten(),
    #         nn.Linear(7 * 7 * 1024, 4096),
    #         nn.LeakyReLU(0.1, inplace=True),
    #         nn.Dropout(0.5, inplace=False), # is it okay to use Dropout with BatchNorm?
    #         nn.Linear(4096, S * S * (5 * B + C)),
    #         nn.Sigmoid()
    #     )
    #
    #     return net