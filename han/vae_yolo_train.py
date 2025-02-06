import torch
import torch.nn as nn
import torch.optim as optim
from vae_yolo import VAE_YOLO  # VAE-YOLO 모델
from dataset import VOCDataset  # 데이터셋 클래스
from tqdm import tqdm
from new_loss import Loss
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import cv2
import matplotlib

matplotlib.use('TkAgg')


# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()
])

def make_folder(base_dir, prefix="train"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    existing_folders = [f for f in os.listdir(base_dir) if f.startswith(prefix)]
    if not existing_folders:
        return os.path.join(base_dir, f"{prefix}1")
    max_index = max([int(f[len(prefix):]) for f in existing_folders if f[len(prefix):].isdigit()], default=0)
    return os.path.join(base_dir, f"{prefix}{max_index + 1}")


def visualize_yolo_output(image, yolo_output, yolo_save_dir, epoch):
    """
    YOLO 출력값을 시각화하는 함수
    """
    filename = f"yolo_epoch_{epoch}.png"
    # 이미지가 PIL Image이면 NumPy array로 변환
    if hasattr(image, "convert"):
        image = np.array(image)

    # 이미지가 RGB 형식이라고 가정하고, OpenCV에서는 BGR로 변환하여 사용
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width, _ = img.shape

    # yolo_output가 텐서라면 NumPy로 변환
    if hasattr(yolo_output, "cpu"):
        yolo_output = yolo_output.cpu().numpy()

    # yolo_output shape: (S, S, 6), 여기서 6 = (x_center, y_center, width, height, confidence, class_prob)
    grid_size = yolo_output.shape[0]
    cell_size = width // grid_size

    # 먼저 각 셀에서 confidence 값을 추출하여 최대 confidence 값을 구함 (옵션)
    confs = []
    for n in range(grid_size):
        for m in range(grid_size):
            # 각 셀의 첫 6개 값: [x, y, w, h, conf, class_prob]
            _, _, _, _, conf, _ = yolo_output[n, m, :6]
            confs.append(conf)
    max_conf = max(confs)
    print("Max confidence:", max_conf)

    # 각 그리드 셀을 순회하며, 최대 confidence 값을 가진 셀의 바운딩 박스를 그리기
    for i in range(grid_size):
        for j in range(grid_size):
            x, y, w, h, conf, cls = yolo_output[i, j, :6]
            # 원하는 기준에 따라, 예를 들어 최대 confidence 값인 셀만 표시하는 경우:
            if conf == max_conf:
                # 실제 이미지 좌표로 변환
                x_center = int((j + x) * cell_size)
                y_center = int((i + y) * cell_size)
                box_w = int(w * width)
                box_h = int(h * height)

                x_min = max(0, x_center - box_w // 2)
                y_min = max(0, y_center - box_h // 2)
                x_max = min(width, x_center + box_w // 2)
                y_max = min(height, y_center + box_h // 2)
                print(f"Bounding Box: {(x_min, y_min, x_max, y_max)} with confidence: {conf:.2f}")

                # 바운딩 박스 그리기
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(img, f"Conf: {conf:.2f}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 이미지 저장: BGR 이미지를 다시 RGB로 변환하여 저장 (plt.imsave는 RGB를 기대함)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    os.makedirs(yolo_save_dir, exist_ok=True)
    save_path = os.path.join(yolo_save_dir, filename)

    # plt.imsave를 사용하여 저장
    plt.imsave(save_path, img_rgb)


def save_test_image_with_bbox(epoch, model, test_image_path, vae_save_dir, yolo_save_dir):

    model.eval()  # 평가 모드로 전환
    model.to(device)
    test_image = Image.open(test_image_path).convert('RGB')  # 이미지를 RGB로 변환
    test_image_tensor = transform(test_image).to(device)  # 전처리 후 GPU/CPU로 이동
    if test_image_tensor.dim() == 3:
        test_image_tensor = test_image_tensor.unsqueeze(0)
    with torch.no_grad():
        recon_x, yolo_output, mu, logvar = model(test_image_tensor)

    # 복원된 이미지: (1, 3, 448, 448) -> (448, 448, 3)
    recon_img = recon_x.squeeze(0).cpu().permute(1, 2, 0).numpy()

    # 저장 디렉터리 생성
    os.makedirs(vae_save_dir, exist_ok=True)
    # 복원 이미지 저장 (PNG 파일)
    recon_save_path = os.path.join(vae_save_dir, f"vae_epoch_{epoch}.png")
    plt.imsave(recon_save_path, recon_img)

        # # **3. 복원된 이미지 출력**
        # recon_img = recon_x.squeeze(0).cpu().permute(1, 2, 0).numpy()
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(test_image_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()))
        # plt.axis("off")
        # plt.title("Original Image")
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(recon_img)
        # plt.axis("off")
        # plt.title("Reconstructed Image")
        #
        # plt.show()
    yolo_output_numpy = yolo_output.squeeze(0).cpu().numpy()
    visualize_yolo_output(test_image, yolo_output_numpy, yolo_save_dir, epoch)


# def loss_function_MSE(recon_x, target_image, mu, logvar, input_image):
#     # MSE = nn.functional.mse_loss(recon_x, y, reduction='sum')
#     # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#     # return MSE + KLD
#     recon_loss = nn.MSELoss()(recon_x, target_image)
#     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / input_image.size(0)
#     return recon_loss + kl_loss

def loss_function_MSE(recon_x, y, mu, logvar, beta=0.1): # ✅ KL Weight 추가
    recon_loss = nn.functional.mse_loss(recon_x, y, reduction='sum')  # MSE 사용
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * beta     # ✅ KL 가중치 조정
    return recon_loss + kl_div


def train_with_bbox_loss(epoch, loss_fn):
    model.train()
    train_loss = 0

    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch + 1}/{epochs}")

    for batch_idx, (input_data, target_data, img, bbox_label) in progress_bar:
        input_data, target_data, img, bbox_label = (
            input_data.to(device),
            target_data.to(device),
            img.to(device),
            bbox_label.to(device)
        )
        optimizer.zero_grad()

        # Forward Pass
        recon_batch, yolo_output, mu, log_var = model(input_data)

        # VAE Loss
        vae_loss = loss_fn(recon_batch, target_data, mu, log_var)
        # Bounding Box Loss
        yolo_loss = criterion(yolo_output, bbox_label)

        # Total Loss
        loss = vae_loss + yolo_loss

        # Backpropagation
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        progress_bar.set_postfix({
            "VAE Loss": f"{vae_loss.item():.4f}",
            "YOLO Loss": f"{yolo_loss.item():.4f}"
        })

    avg_loss = train_loss / len(data_loader.dataset)

    print(f"Epoch-{epoch + 1}/{epochs}, Total Loss: {avg_loss:.4f}")
    return train_loss / len(data_loader.dataset)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.set_printoptions(profile="full")

    input_dir = 'data/1000/yes_sfx/'
    target_dir = 'data/1000/no_sfx/'

    train_label = ('data/1000/bbox.txt')

    vae_test_results = 'data/1000/results/vae'
    yolo_test_results = 'data/1000/results/yolo'
    test_image = 'data/1000/yes_sfx/image_1.jpeg'

    batch_size = 32
    learning_rate = 1e-4
    epochs = 10
    latent_dim = 100
    grid_size = 7
    bbox_attrs = 2
    num_classes = 1

    criterion = Loss(feature_size=grid_size)

    model = VAE_YOLO(input_channels=3, latent_dim=latent_dim, grid_size=grid_size, bbox_atts=bbox_attrs,
                     num_classes=num_classes).to(device)

    dataset = VOCDataset(False, input_dir, target_dir, input_dir, train_label)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = loss_function_MSE
    for epoch in range(epochs):
        avg_train_loss = train_with_bbox_loss(epoch, loss_fn)
        save_test_image_with_bbox(epoch, model, test_image, vae_test_results, yolo_test_results)



