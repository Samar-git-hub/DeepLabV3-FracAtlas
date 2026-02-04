import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
from medpy.metric.binary import hd95
from sklearn.metrics import roc_auc_score
from thop import profile
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")
root_dir = 'data/FracAtlas'
test_csv_path = 'data/FracAtlas/processed/original_1024/test.csv'
processed_dir = 'data/FracAtlas/processed/original_1024/test'
orig_res_mask_dir = 'data/FracAtlas/masks/Fractured'
target_size = 1024

def get_inverse_transform_params(orig_h, orig_w, target_size=1024):
    
    scale = target_size / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)

    # Total padding needed, in accordance with albumentations 'pad' source code
    pad_h = target_size - new_h
    pad_w = target_size - new_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return new_h, new_w, pad_top, pad_bottom, pad_left, pad_right

def evaluate_model(model, model_name="TestModel"):
    
    model.eval()
    model.to(device)

    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")
    
    test_df = pd.read_csv(test_csv_path)

    # Static metrics
    print(f"Calculating GFLOPS and Parameters for {model_name}")
    dummy_input = torch.randn(1, 3, target_size, target_size).to(device)
    
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9
        print(f"GFLOPS: {gflops:.4f} | Parameters: {params:,.0f}")
    except Exception as e:
        print(f"GFLOPS failed: {e}")
        gflops, params = 0, 0

    # Dynamic metrics 
    total_iou = []
    total_dice = []
    total_acc = []
    total_hd95 = []
    total_map50 = []

    # Global pools for AUC (Micro-Average)
    global_probs = []
    global_targets = []

    transform_input = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    print(f"Starting evaluation on {len(test_df)} images")

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):

            img_name = row['image_id']
            orig_source = row['original_source']
            
            # loading processed image
            proc_img_path = os.path.join(processed_dir, 'images', img_name)

            # for measuring IoU, Dice and more, on the original resolution images
            orig_res_name = os.path.splitext(orig_source)[0] + '.png'
            orig_res_path = os.path.join(orig_res_mask_dir, orig_res_name)

            if not os.path.exists(proc_img_path) or not os.path.exists(orig_res_path):
                continue

            img = cv2.imread(proc_img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask_orig = cv2.imread(orig_res_path, cv2.IMREAD_GRAYSCALE)
            mask_orig = (mask_orig > 127).astype(np.uint8)

            orig_h, orig_w = mask_orig.shape[:2]

            input_tensor = transform_input(image=img)['image'].unsqueeze(0).to(device)

            # Inference 
            logits = model(input_tensor)

            if isinstance(logits, dict):
                logits = logits['out']

            # Probabilities (for AUC)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

            # Inverse transform and upsampling
            new_h, new_w, p_top, p_btm, p_l, p_r = get_inverse_transform_params(orig_h, orig_w, target_size)  

            crop_h = target_size - p_btm
            crop_w = target_size - p_r
            probs_cropped = probs[p_top:crop_h, p_l:crop_w]

            probs_orig_res = cv2.resize(probs_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            global_probs.append(probs_orig_res.flatten())
            global_targets.append(mask_orig.flatten())

            pred_orig_res = (probs_orig_res > 0.1).astype(np.uint8)

            # IoU
            intersection = np.sum(pred_orig_res * mask_orig)
            pred_area = np.sum(pred_orig_res)
            gt_area = np.sum(mask_orig)
            union = pred_area + gt_area - intersection

            iou = (intersection + 1e-6) / (union + 1e-6)
            total_iou.append(iou)

            # mAP @ IoU = 0.5
            total_map50.append(1.0 if iou >= 0.5 else 0.0)

            # Dice
            dice = (2 * intersection + 1e-6) / (pred_area + gt_area + 1e-6)
            total_dice.append(dice)

            # Accuracy
            acc = np.sum(pred_orig_res == mask_orig) / (orig_h * orig_w)
            total_acc.append(acc)

            # HD95
            if pred_area > 0 and gt_area > 0:
                
                val_hd95 = hd95(pred_orig_res, mask_orig, voxelspacing=None)
                total_hd95.append(val_hd95)

            elif pred_area == 0 and gt_area == 0:

                total_hd95.append(0.0)

            else:

                total_hd95.append(np.sqrt(orig_h**2 + orig_w**2))

            del probs_orig_res, pred_orig_res, probs_cropped
        
        # AUC calculation
        try:

            all_probs = np.concatenate(global_probs)
            all_targets = np.concatenate(global_targets)

            del global_probs, global_targets

            global_auc = roc_auc_score(all_targets, all_probs)
        
        except Exception as e:
            print(f"AUC calculation failed: {e}")
            global_auc = 0.0

        
        results = {
            "Model": model_name,
            "IoU": np.mean(total_iou),
            "mAP@0.5": np.mean(total_map50),
            "Dice": np.mean(total_dice),
            "HD95": np.mean(total_hd95),
            "Accuracy": np.mean(total_acc),
            "AUC": global_auc,
            "GFLOPS": gflops,
            "Params": params
        }

        print(f"Final Results for {model_name}:")
        print(f"IoU:                {results['IoU']:.4f}")
        print(f"mAP @ IoU = 0.5:    {results['mAP@0.5']:.4f}")
        print(f"Dice:               {results['Dice']:.4f}")
        print(f"HD95:               {results['HD95']:.2f} px")
        print(f"Accuracy:           {results['Accuracy']:.4f}")
        print(f"AUC:                {results['AUC']:.4f}")
        print(f"GFLOPS:             {results['GFLOPS']:.4f}")
        print(f"Params:             {results['Params']:,.0f}")

if __name__ == "__main__":
    pass