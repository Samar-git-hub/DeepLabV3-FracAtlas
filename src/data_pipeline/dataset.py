import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class FracAtlasPipeline(Dataset):
    def __init__(self, root_dir='data/FracAtlas', split='train', mode='original_resized', resolution=512):
        self.root_dir = root_dir
        self.split = split
        self.mode = mode

        if resolution == 1024:
            suffix = "_1024"
        else:
            suffix = ""

        if self.mode == 'augmented':
            self.data_dir = os.path.join(root_dir, f'Augmented{suffix}')
        else:
            self.data_dir = os.path.join(root_dir, 'processed', f'original{suffix}')

        csv_path = os.path.join(self.data_dir, f'{split}.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at {csv_path}. Create the CSV using augment.py with target_size={resolution}")
        
        self.df = pd.read_csv(csv_path)

        # Normalizing images and converting them to Tensors, in order to feed them to the segmentation models
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_id']
        mask_name = row['mask_file']

        img_path = os.path.join(self.data_dir, self.split, 'images', img_name)
        mask_path = os.path.join(self.data_dir, self.split, 'masks', mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)

        # Image formatted for the input, mask formatted for loss function
        return {
            'image': augmented['image'],
            'mask':  augmented['mask'].float().unsqueeze(0),
            'img_name': img_name
        }