import os
import random
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class VOCDataset(Dataset):

    def __init__(self, is_train, input_dir, target_dir, image_dir, label_txt, image_size=448, grid_size=7, num_bboxes=2,
                 num_classes=1):
        self.is_train = is_train
        self.image_size = image_size

        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

        # VAE를 위한 전처리
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_names = os.listdir(input_dir)
        self.target_names = os.listdir(target_dir)

        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        self.mean = np.array(mean_rgb, dtype=np.float32)

        self.to_tensor = transforms.ToTensor()

        if isinstance(label_txt, list) or isinstance(label_txt, tuple):
            # cat multiple list files together.
            # This is useful for VOC2007/VOC2012 combination.
            tmp_file = '/tmp/label.txt'
            os.system('cat %s > %s' % (' '.join(label_txt), tmp_file))
            label_txt = tmp_file

        self.paths, self.boxes, self.labels = [], [], []

        with open(label_txt) as f:
            lines = f.readlines()

        for line in lines:
            splitted = line.strip().split()

            fname = splitted[0]
            path = os.path.join(image_dir, fname)
            self.paths.append(path)

            num_boxes = (len(splitted) - 1) // 5
            box, label = [], []
            for i in range(num_boxes):
                x1 = float(splitted[5 * i + 1])
                y1 = float(splitted[5 * i + 2])
                x2 = float(splitted[5 * i + 3])
                y2 = float(splitted[5 * i + 4])
                c = int(splitted[5 * i + 5])
                box.append([x1, y1, x2, y2])
                label.append(c)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

        self.num_samples = len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        boxes = self.boxes[idx].clone()  # [n, 4]
        labels = self.labels[idx].clone()  # [n,]

        # VAE를 위한 전처리
        input_name = os.path.join(self.input_dir, self.input_names[idx])
        target_name = os.path.join(self.target_dir, self.target_names[idx])
        input_image = Image.open(input_name).convert('RGB')  # 이미지를 RGB 형식으로 변환
        target_image = Image.open(target_name).convert('RGB')
        input_image = transforms.Resize((448, 448))(input_image)
        target_image = transforms.Resize((448, 448))(target_image)
        input_image = transforms.ToTensor()(input_image)
        target_image = transforms.ToTensor()(target_image)

        # For debug.
        debug_dir = 'tmp/voc_tta'
        os.makedirs(debug_dir, exist_ok=True)
        img_show = img.copy()
        box_show = boxes.numpy().reshape(-1)
        n = len(box_show) // 4
        for b in range(n):
            pt1 = (int(box_show[4 * b + 0]), int(box_show[4 * b + 1]))
            pt2 = (int(box_show[4 * b + 2]), int(box_show[4 * b + 3]))
            cv2.rectangle(img_show, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=1)
        cv2.imwrite(os.path.join(debug_dir, 'test_{}.jpg'.format(idx)), img_show)

        h, w, _ = img.shape
        boxes /= torch.Tensor([[w, h, w, h]]).expand_as(boxes)  # normalize (x1, y1, x2, y2) w.r.t. image width/height.
        target = self.encode(boxes, labels)  # [S, S, 5 x B + C]

        img = cv2.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # assuming the model is pretrained with RGB images.
        img = (img - self.mean) / 255.0  # normalize from -1.0 to 1.0.
        img = self.to_tensor(img)

        return input_image, target_image, img, target

    def __len__(self):
        return self.num_samples

    def encode(self, boxes, labels):
        """ Encode box coordinates and class labels as one target tensor.
        Args:
            boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...], normalized from 0.0 to 1.0 w.r.t. image width/height.
            labels: (tensor) [c_obj1, c_obj2, ...]
        Returns:
            An encoded tensor sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
        """

        S, B, C = self.S, self.B, self.C
        N = 5 * B + C

        target = torch.zeros(S, S, N)
        cell_size = 1.0 / float(S)
        boxes_wh = boxes[:, 2:] - boxes[:, :2]  # width and height for each box, [n, 2]
        boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0  # center x & y for each box, [n, 2]
        for b in range(boxes.size(0)):
            xy, wh, label = boxes_xy[b], boxes_wh[b], int(labels[b])

            ij = (xy / cell_size).ceil() - 1.0
            i, j = int(ij[0]), int(ij[1])  # y & x index which represents its location on the grid.
            x0y0 = ij * cell_size  # x & y of the cell left-top corner.
            xy_normalized = (xy - x0y0) / cell_size  # x & y of the box on the cell, normalized from 0.0 to 1.0.

            # TBM, remove redundant dimensions from target tensor.
            # To remove these, loss implementation also has to be modified.
            for k in range(B):
                s = 5 * k
                target[j, i, s:s + 2] = xy_normalized
                target[j, i, s + 2:s + 4] = wh
                target[j, i, s + 4] = 1.0
            target[j, i, 5 * B + label] = 1.0

        return target


def test():
    from torch.utils.data import DataLoader

    image_dir = 'data/1000/yes_sfx/'
    label_txt = ['data/1000/bbox.txt']

    dataset = VOCDataset(False, image_dir, label_txt)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    data_iter = iter(data_loader)
    for i in range(100):
        img, target = next(data_iter)
        print(img.size(), target.size())


if __name__ == '__main__':
    test()