import albumentations as A
import cv2
import numpy as np
import os 
import pandas as pd
import json
from tqdm import tqdm

root_dir = 'data/FracAtlas'

multiplier = 50
target_size = 1024

if target_size == 1024:
    aug_output_dir = 'data/FracAtlas/Augmented_1024'
    orig_resized_dir = 'data/FracAtlas/processed/original_1024'
else:
    aug_output_dir = 'data/FracAtlas/Augmented'
    orig_resized_dir = 'data/FracAtlas/processed/original'

print(f"Target Resolution: {target_size}x{target_size}")

aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.GaussianBlur(p=0.5),
    A.RandomGamma(p=0.5),

    # Resize preserving aspect ratio
    A.LongestMaxSize(max_size=target_size),
    A.PadIfNeeded(min_height=target_size, min_width=target_size, border_mode=cv2.BORDER_CONSTANT, position='center')
])

resize_pipeline = A.Compose([
    A.LongestMaxSize(max_size=target_size),
    A.PadIfNeeded(min_height=target_size, min_width=target_size, border_mode=cv2.BORDER_CONSTANT, position='center')
])

def load_coco_data(root_dir):

    # Loading annotations (for creating a mask)
    json_path = os.path.join(root_dir, 'Annotations', 'COCO JSON', 'COCO_fracture_masks.json')

    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Filename lookup
    img_id_map = { img['id']: img['file_name'] for img in coco_data['images']}

    # Coordinates lookup
    ann_map = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']

        if img_id not in ann_map:
            ann_map[img_id] = []
        
        ann_map[img_id].append(ann['segmentation'])

    return img_id_map, ann_map
        
def create_dataset(split_name, mode='original_resized'):
    # 'augmented' for 50 data augmentation iterations for each image in the train split
    # 'original_resized' fo resizing the train, validation and test splits
    
    print(f"Processing {split_name} ({mode})")

    csv_path = os.path.join(root_dir, 'Utilities', 'Fracture Split', f"{split_name}.csv")
    if not os.path.exists(csv_path):
        print("CSV not found")
        return
    
    df = pd.read_csv(csv_path)
    id_to_file, id_to_anns = load_coco_data(root_dir)

    if mode == 'augmented':
        base_out = aug_output_dir
        current_multiplier = multiplier
        pipeline = aug_pipeline
    else:
        base_out = orig_resized_dir
        current_multiplier = 1
        pipeline = resize_pipeline

    csv_save_path = os.path.join(base_out, f'{split_name}.csv')
    if os.path.exists(csv_save_path):
        print(f"Skipping {split_name} ({mode}): Output CSV already exists")
        return
    
    split_out_dir = os.path.join(base_out, split_name)
    img_out_dir = os.path.join(split_out_dir, 'images')
    mask_out_dir = os.path.join(split_out_dir, 'masks')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    new_csv_rows = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['image_id']

        img_id_numeric = None
        for k, v in id_to_file.items():
            if v == filename:
                img_id_numeric = k
                break
        
        if img_id_numeric is None: 
            continue

        img_path = os.path.join(root_dir, 'images', 'Fractured', filename)
        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Creating high-resolution mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if img_id_numeric in id_to_anns:
            for pols in id_to_anns[img_id_numeric]:
                for poly in pols:
                    pts = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)

        
        # Generation
        for i in range(current_multiplier):
            transformed = pipeline(image=image, mask=mask)
            trans_img = transformed['image']
            trans_mask = transformed['mask']
            trans_mask = (trans_mask > 0.5).astype(np.uint8)

            if mode == 'augmented':
                new_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
                new_maskname = f"{os.path.splitext(filename)[0]}_aug_{i}.png"
            else:
                new_filename = filename
                new_maskname = f"{os.path.splitext(filename)[0]}.png"
            
            save_img_path = os.path.join(img_out_dir, new_filename)
            save_mask_path = os.path.join(mask_out_dir, new_maskname)

            # Save
            print(f"Saving {mode} images and masks")
            cv2.imwrite(save_img_path, cv2.cvtColor(trans_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_mask_path, trans_mask * 255)

            new_csv_rows.append({
                'image_id': new_filename,
                'mask_file': new_maskname,
                'original_source': filename
            })
    
    pd.DataFrame(new_csv_rows).to_csv(csv_save_path, index=False)
    print(f"Saved CSV to {csv_save_path}")

def generate_original_masks():
    print("Generating Original Resolution Masks")

    mask_out_dir = os.path.join(root_dir, 'masks', 'Fractured')
    os.makedirs(mask_out_dir, exist_ok=True)

    id_to_file, id_to_anns = load_coco_data(root_dir)

    for img_id, filename in tqdm(id_to_file.items(), total=len(id_to_file)):

        save_path = os.path.join(mask_out_dir, f"{os.path.splitext(filename)[0]}.png")
        if os.path.exists(save_path):
            continue

        img_path = os.path.join(root_dir, 'images', 'Fractured', filename)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)

        if img_id in id_to_anns:
            for pols in id_to_anns[img_id]:
                for poly in pols:
                    pts = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
        
        cv2.imwrite(save_path, mask * 255)

if __name__ == "__main__":

    create_dataset('train', mode='augmented')
    create_dataset('train', mode='original_resized')
    create_dataset('valid', mode='original_resized')
    create_dataset('test',  mode='original_resized')
    generate_original_masks()