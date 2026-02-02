import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline.dataset import FracAtlasPipeline
from src.models.deeplabv3_mobilenetv3 import get_deeplab_model

config = {
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "root_dir": 'data/FracAtlas',
    "save_dir": 'experiments/Exp6_DeepLab_MobileNetV3_ImageNet_1024',
    "epochs": 50,
    "resolution": 1024,
    "batch_size": 16,
    "learning_rate": 0.001,
    "num_workers": 2,
    "lr_factor": 0.9,
    "lr_patience": 5
}

print(f"Using {config['device']} | Resolution: {config['resolution']} | Batch Size: {config['batch_size']}")

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice

def train_one_epoch(model, loader, optimizer, criterion_bce, criterion_dice, device):
    model.train()
    running_loss = 0.0

    loop = tqdm(loader, desc="Training", leave=False)

    for batch in loop:
        
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        logits = outputs['out']
        aux_logits = outputs['aux']

        # Main Loss
        loss_bce_main = criterion_bce(logits, masks)
        # loss_dice_main = criterion_dice(logits, masks)
        # loss_main = 0.5 * loss_bce_main + 0.5 * loss_dice_main
        loss_main = loss_bce_main

        # Auxiliary Loss
        loss_bce_aux = criterion_bce(aux_logits, masks)
        # loss_dice_aux = criterion_dice(aux_logits, masks)
        # loss_aux = 0.5 * loss_bce_aux + 0.5 * loss_dice_aux
        loss_aux = loss_bce_aux

        # Auxiliary loss toggle
        loss = loss_main + (0.5 * loss_aux)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    return running_loss / len(loader)

def validate(model, loader, criterion_bce, criterion_dice, device):
    
    model.eval()
    running_loss = 0.0
    total_iou_score = 0.0
    total_images = 0

    with torch.no_grad():

        for batch in loader:
            
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            logits = outputs['out']

            loss_bce = criterion_bce(logits, masks)
            # loss_dice = criterion_dice(logits, masks)
            # loss = 0.5 * loss_bce + 0.5 * loss_dice
            loss = loss_bce

            running_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            preds_flat = preds.view(preds.size(0), -1)
            masks_flat = masks.view(masks.size(0), -1)

            intersection = (preds_flat * masks_flat).sum(dim=1)
            union = preds_flat.sum(dim=1) + masks_flat.sum(dim=1) - intersection

            batch_ious = (intersection + 1e-6) / (union + 1e-6)

            total_iou_score += batch_ious.sum().item()
            total_images += images.size(0)

    avg_loss = running_loss / len(loader)
    avg_iou = total_iou_score / total_images

    return avg_loss, avg_iou

def main():
    
    os.makedirs(config['save_dir'], exist_ok=True)

    print(f"Initializing data using 'augmented' data")
    train_dataset = FracAtlasPipeline(split='train', mode='augmented', resolution=config['resolution'])
    valid_dataset = FracAtlasPipeline(split='valid', mode='original_resized', resolution=config['resolution'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'], pin_memory=True)

    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'],
                              shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    model = get_deeplab_model(config['device'], model_weights = "ImageNet")
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['lr_factor'], patience=config['lr_patience']
    )

    best_iou = 0.0
    start_time = time.time()

    log_path = os.path.join(config['save_dir'], 'training_log.csv')
    if not os.path.exists(log_path):
        pd.DataFrame(columns=['Epoch', 'Train_Loss', 'Val_Loss', 'Val_IoU', 'LR', 'Time']).to_csv(log_path, index=False)
    
    print(f"Training for {config['epochs']} epochs")
    
    for epoch in range(1, config['epochs'] + 1):
        
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_bce, criterion_dice, config['device'])
        val_loss, val_iou = validate(model, valid_loader,  criterion_bce, criterion_dice, config['device'])
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if val_iou > best_iou:
            best_iou = val_iou
            save_path = os.path.join(config['save_dir'], "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best average validation IoU: {best_iou:.4f}")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{config['epochs']} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss} | Validation IoU: {val_iou} | Time: {epoch_time:.0f}s")

        new_row = pd.DataFrame([{
            'Epoch': epoch,
            'Train_Loss': train_loss,
            'Val_Loss': val_loss,
            'Val_IoU': val_iou,
            'LR': current_lr,
            'Time': epoch_time
        }])
        new_row.to_csv(log_path, mode='a', header=False, index=False)
    
    total_time = (time.time() - start_time) / 60
    print(f"\nTraining Complete in {total_time:.2f} minutes")
    print(f"Best Validation Avg IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()