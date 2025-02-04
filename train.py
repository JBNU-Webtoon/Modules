import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from VAE.vae_detection_yo_modified import VAE, loss_function_MSE
from VAE.voc import VOCDataset
from VAE.loss import Loss
from CRNN.dataset import Synth90kDataset, synth90k_collate_fn
from CRNN.model import CRNN
import matplotlib.pyplot as plt
import numpy as np


# Define combined model with feature fusion
class CombinedModel(nn.Module):
    def __init__(self, vae, crnn, latent_dim):
        super(CombinedModel, self).__init__()
        self.vae = vae
        self.crnn_cnn = crnn.cnn  # CRNN의 CNN 계층
        self.latent_dim = latent_dim

        # Linear layer to match dimensions for fusion
        self.fc_fusion = nn.Linear(2*3*512*16*16, 64)  # Adjust the output size as needed

    def forward(self, webtoon_image, text_image):
        # VAE forward pass
        h, recon, mu, log_var, detection_output = self.vae(webtoon_image)

        print("h: ", h.shape)

        # CRNN CNN pass (text image features)
        crnn_features = self.crnn_cnn(text_image)  # CNN 계층에서 특징 추출
        print("crnn_features: ", crnn_features.shape)
        crnn_features = crnn_features.view(crnn_features.size(0), -1)  # Flatten
        print("crnn_features: ", crnn_features.shape)

        return recon, mu, log_var, detection_output, h, crnn_features
    

# Hyperparameters
LATENT_DIM = 100
IMG_HEIGHT, IMG_WIDTH = 32, 100
NUM_CLASSES = len(Synth90kDataset.CHAR2LABEL) + 1  # Including blank for CTC

# Initialize models
vae = VAE(input_channels=3, hidden_dim=400, latent_dim=LATENT_DIM).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
crnn = CRNN(img_channel=1, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_class=NUM_CLASSES).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model = CombinedModel(vae, crnn, LATENT_DIM).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loss functions
vae_loss_fn = loss_function_MSE
detection_loss_fn = Loss(        # YOLO 기반 손실
    feature_size=8,              # Feature map 크기
    num_bboxes=2,                # 바운딩 박스 수
    num_classes=2,               # 클래스 수
    lambda_coord=5.0,            # 위치/크기 손실 가중치
    lambda_noobj=0.5             # No-object 손실 가중치
)
ctc_loss_fn = nn.CTCLoss(reduction='sum', zero_infinity=True)

# Datasets and DataLoaders
vae_dataset = VOCDataset(
    is_train=False,
    input_dir="./data/Images_with_effects",
    target_dir="./data/Images",
    image_dir="./data/Images_with_effects",
    label_txt="./data/bbox.txt"
)
vae_loader = DataLoader(vae_dataset, batch_size=8, shuffle=True, num_workers=4)

crnn_dataset = Synth90kDataset(root_dir="./data", mode="train")
crnn_loader = DataLoader(crnn_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=synth90k_collate_fn)

def loss_norm(e):
    n = int(torch.log10(e))
    result = e / 10**(n+1)
    return result

# Training loop
def train_model(num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for (vae_data, crnn_data) in zip(vae_loader, crnn_loader):
            # VAE Data (webtoon images)
            webtoon_image, target_data, _, bbox_label = vae_data
            webtoon_image, target_data, bbox_label = (
                webtoon_image.to(torch.device("cuda")),
                target_data.to(torch.device("cuda")),
                bbox_label.to(torch.device("cuda")),
            )

            # CRNN Data (text images)
            text_image, targets, target_lengths = crnn_data
            text_image, targets, target_lengths = (
                text_image.to(torch.device("cuda")),
                targets.to(torch.device("cuda")),
                target_lengths.to(torch.device("cuda")),
            )
            #target_lengths = torch.flatten(target_lengths)
            target_lengths = target_lengths.view(-1)
            print("target_lengths: ", target_lengths.shape)

            # Forward pass
            recon, mu, log_var, detection_output, h, crnn_features = model(webtoon_image, text_image)

            text_output = crnn(text_image)

            # Compute losses
            vae_loss = vae_loss_fn(recon, target_data, mu, log_var)
            det_loss = detection_loss_fn(detection_output, bbox_label)
            # text_output 차원 변경
            # text_output = text_output.permute(1, 0, 2)  # (batch_size, seq_length, num_classes) -> (seq_length, batch_size, num_classes)

            # Compute CTC Loss
            batch_size = text_output.size(1)
            sequence_length = text_output.size(0)
            input_lengths = torch.full((batch_size,), sequence_length, dtype=torch.long).to(text_output.device)
            # 크기 출력하여 디버깅
            print("batch_size: ", batch_size)

            # text_output에서 각 타임스텝에 대해 가장 높은 확률을 가진 인덱스 추출
            predicted_indices = torch.argmax(text_output, dim=2)  # (24, 8) 크기


            # 예측된 인덱스 시퀀스
            predicted_texts = []
            start_idx = 0
            for i in range(len(target_lengths)):
                end_idx = start_idx + target_lengths[i]
                predicted_text = predicted_indices[:target_lengths[i], i]  # i번째 배치에 대한 예측 텍스트
                predicted_texts.append(predicted_text)
                start_idx = end_idx

            # 각 배치에 대한 예측된 텍스트 출력
            for i, pred in enumerate(predicted_texts):
                print(f"Batch {i + 1} Predicted Text: {pred}")

            print(f"text_output shape: {text_output.shape}")  # [seq_length, batch_size, num_classes]
            print("target: ", targets)
            print(f"targets shape: {targets.shape}")
            print("input_lengths: ", input_lengths)
            print(f"input_lengths shape: {input_lengths.shape}")
            print("target_lengths: ", target_lengths)
            print(f"target_lengths shape: {target_lengths.shape}")
            ctc_loss = ctc_loss_fn(text_output, targets, input_lengths, target_lengths)

            
            fusion_loss = nn.functional.mse_loss(h, crnn_features)
            # lambda_vae = 1
            # lambda_det = 1
            # lambda_ctc = 0.1
            # lambda_fus = 0.001
            total_batch_loss = loss_norm(vae_loss) + loss_norm(det_loss) + loss_norm(ctc_loss) + loss_norm(fusion_loss)
            print("vae loss: ", vae_loss, loss_norm(vae_loss))
            print("det_loss: ", det_loss, loss_norm(det_loss))
            print("ctc_loss: ", ctc_loss, loss_norm(ctc_loss))
            print("fusion_loss: ", fusion_loss, loss_norm(fusion_loss))
            print("total_batch_loss: ", total_batch_loss)
            # Backward pass
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

        # Calculate total number of batches from both loaders
        num_batches = min(len(vae_loader), len(crnn_loader))

        # Average loss calculation
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / num_batches:.4f}")


# Train the combined model
train_model(num_epochs=10)
