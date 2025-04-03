
import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms.functional import resize as F_resize

class HandSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, size=(256,256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_names = sorted(os.listdir(images_dir))
        self.transform = transform
        self.target_size = size

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = img_name.replace(".jpg", "_mask.png")
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Open images
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")
        image = F_resize(image, self.target_size)           # bilinear by default
        mask  = F_resize(mask,  self.target_size)

        if self.transform is not None:
            image = self.transform(image)

        # Convert mask to tensor 0 or 1
        mask_np = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask_np)
        # threshold
        mask_tensor = (mask_tensor > 127).long()  # 0 or 1
        # add channel dimension => (1,H,W)
        mask_tensor = mask_tensor.unsqueeze(0)

        return image, mask_tensor
