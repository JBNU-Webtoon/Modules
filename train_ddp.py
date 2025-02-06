import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from VAE.vae_detection_yo_modified import VAE, loss_function_MSE
from VAE.voc import VOCDataset
from VAE.loss import Loss
from CRNN.dataset import Synth90kDataset, synth90k_collate_fn
from CRNN.model import CRNN

def setup(rank, world_size):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'  # 🔹 GPU 1,2,3,4번만 사용
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group() 

class CombinedModel(nn.Module):
    def __init__(self, vae, crnn, latent_dim):
        super(CombinedModel, self).__init__()
        self.vae = vae
        self.crnn_cnn = crnn.cnn  # CRNN의 CNN 계층
        self.latent_dim = latent_dim

    def forward(self, webtoon_image, text_image):
        h, recon, mu, log_var, detection_output = self.vae(webtoon_image)
        crnn_features = self.crnn_cnn(text_image).view(text_image.size(0), -1)
        return recon, mu, log_var, detection_output, h, crnn_features

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())  # 바이트 단위
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Total Memory Size: {total_size / (1024 ** 2):.2f} MB")  # MB 단위로 변환

def train(rank, world_size, num_epochs=10):
    setup(rank, world_size)

    print("Available CUDA Devices:", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    device = torch.device(f"cuda:{rank}")
    print("device: ", device)
    vae = VAE(input_channels=3, hidden_dim=400, latent_dim=512).to(device)
    crnn = CRNN(img_channel=1, img_height=32, img_width=100, num_class=38).to(device)
    model = CombinedModel(vae, crnn, 512).to(device)
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        print_model_size(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    vae_loss_fn = loss_function_MSE
    detection_loss_fn = Loss(feature_size=8, num_bboxes=2, num_classes=2, lambda_coord=5.0, lambda_noobj=0.5)
    ctc_loss_fn = nn.CTCLoss(reduction='sum', zero_infinity=True)

    vae_dataset = VOCDataset(is_train=False, input_dir="../Modules/data/Images_with_effects", target_dir="/home/jbnu/Images", image_dir="../Modules/data/Images_with_effects", label_txt="../Modules/data/bbox.txt")
    crnn_dataset = Synth90kDataset(root_dir="../Modules/data", mode="train")
    
    vae_sampler = DistributedSampler(vae_dataset, num_replicas=world_size, rank=rank, drop_last=True)
    crnn_sampler = DistributedSampler(crnn_dataset, num_replicas=world_size, rank=rank)

    vae_loader = DataLoader(vae_dataset, batch_size=8, sampler=vae_sampler, num_workers=0, shuffle=False)
    crnn_loader = DataLoader(crnn_dataset, batch_size=8, sampler=crnn_sampler, num_workers=0, collate_fn=synth90k_collate_fn)

    for epoch in range(num_epochs):
        print(f'epoch{epoch+1} start!')
        vae_sampler.set_epoch(epoch)
        crnn_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        for (vae_data, crnn_data) in zip(vae_loader, crnn_loader):
            print("Before calculating loss")
            webtoon_image, target_data, _, bbox_label = [x.to(device) for x in vae_data]
            text_image, targets, target_lengths = [x.to(device) for x in crnn_data]
            target_lengths = target_lengths.view(-1)

            optimizer.zero_grad()
            recon, mu, log_var, detection_output, h, crnn_features = model(webtoon_image, text_image)
            text_output = crnn(text_image)

            vae_loss = vae_loss_fn(recon, target_data, mu, log_var)
            det_loss = detection_loss_fn(detection_output, bbox_label)
            input_lengths = torch.full((text_output.size(1),), text_output.size(0), dtype=torch.long).to(device)
            ctc_loss = ctc_loss_fn(text_output, targets, input_lengths, target_lengths)
            fusion_loss = nn.functional.mse_loss(h, crnn_features)
            
            print("After calculating loss")
            total_batch_loss = vae_loss + det_loss + ctc_loss + fusion_loss
            print("vae, det, ctc, fusion: ", vae_loss, det_loss, ctc_loss, fusion_loss)
            print("total_batch_loss: ", total_batch_loss)
            total_batch_loss.backward()
            print("After Backpropagation")
            optimizer.step()

            total_loss += total_batch_loss.item()
        
        print(f"Rank {rank}, Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")
    
    cleanup()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

