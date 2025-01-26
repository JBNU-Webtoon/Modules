import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#데이터셋 클래스 정의
class WebtoonDataset(Dataset):
  def __init__(self, input_dir, target_dir, transforms=None):
    self.input_dir = input_dir
    self.target_dir = target_dir
    self.transforms = transforms
    self.input_names = os.listdir(input_dir)
    self.target_names = os.listdir(target_dir)

  def __len__(self):
    return len(self.input_names)

  def __getitem__(self, idx):
    input_name = os.path.join(self.input_dir, self.input_names[idx])
    target_name = os.path.join(self.target_dir, self.target_names[idx])
    input_image = Image.open(input_name).convert('RGB') # 이미지를 RGB 형식으로 변환
    target_image = Image.open(target_name).convert('RGB')
    if self.transforms:
      input_image = self.transforms(input_image)
      target_image = self.transforms(target_image)
    return input_image, target_image