# DeepLabV3 + MobileNetV3 for Bone Fracture Segmentation

An open-source implementation of **DeepLabV3** with a **MobileNetV3-Large** backbone for semantic segmentation of bone fractures in X-ray images, trained and evaluated on the **FracAtlas** dataset.

Inspired by the Elsevier paper:

> **"The impact of implementing backbone architectures on fracture segmentation in X-ray images"**  
> Turk et al., 2024  
> DOI: [10.1016/j.jestch.2024.101883](https://doi.org/10.1016/j.jestch.2024.101883)

This repository implements the **MobileNetV3-Large backbone** variant, focusing on lightweight and efficient fracture segmentation.

---

## Task Overview

**Semantic segmentation** of bone fractures: given an X-ray image, the model produces a pixel-level binary mask highlighting the fracture region. This is a challenging task because fractures are thin, irregular, low-contrast structures that occupy a very small fraction of the image.

<p align="center">
  <em>Input: X-ray image → Output: Binary fracture mask</em>
</p>

---

## Architecture

| Component | Details |
|---|---|
| **Segmentation Model** | DeepLabV3 (Atrous Spatial Pyramid Pooling) |
| **Backbone** | MobileNetV3-Large (ImageNet pretrained) |
| **Output Head** | Modified classifier: 256 → 1 channel (binary segmentation) |
| **Auxiliary Head** | Modified aux classifier: 1 channel output |
| **Loss Function** | 0.5 × BCE + 0.5 × Dice Loss (main) + 0.5 × Auxiliary Loss |
| **Input Resolution** | 1024 × 1024 (aspect-ratio preserved with center padding) |

---

## Dataset

**FracAtlas** — a musculoskeletal X-ray dataset containing 4,083 images (717 fractured, 3,366 non-fractured) covering fractures across arms, legs, shoulders, and hip regions. COCO-format polygon annotations are used to generate binary segmentation masks.

Only the **fractured images** (with segmentation annotations) are used:

| Split | Images |
|---|---|
| Train | 574 |
| Validation | 82 |
| Test | 61 |
| **Total** | **717** |

The splits follow the official FracAtlas `Fracture Split` partition.

---

## Results

Evaluated on the **61-image test set** at original resolution (predictions are inverse-transformed from 1024×1024 back to original image dimensions before computing metrics).

### Test Set Performance

| Metric | Value |
|---|---|
| **IoU** | 0.2871 |
| **Dice Coefficient** | 0.3772 |
| **Pixel Accuracy** | 0.9968 |
| **AUC (ROC)** | 0.9521 |
| **HD95** | 125.51 px |

### Model Efficiency

| Metric | Value |
|---|---|
| **GFLOPs** | 39.83 |
| **Parameters** | 11.02M |

### Training Summary

| Detail | Value |
|---|---|
| **Epochs** | 50 |
| **Best Validation IoU** | 0.337 (Epoch 29) |
| **Optimizer** | Adam (lr=0.001) |
| **LR Schedule** | ReduceLROnPlateau (factor=0.9, patience=5) |
| **Training Data** | Original resized (no augmentation) |
| **Time per Epoch** | ~87s |

### Notes on Results

- **High pixel accuracy (0.9968)** is expected because fractures occupy a very small fraction of the image — even predicting all-background achieves high accuracy. This metric is not informative for this task.
- **AUC of 0.9521** indicates strong discriminative ability at the pixel level — the model assigns higher probabilities to fracture pixels than background pixels.
- **IoU of 0.2871** reflects the difficulty of precisely localizing thin, irregular fracture lines with a lightweight backbone on a small dataset (574 training images, no augmentation).
- **MobileNetV3-Large** is a lightweight backbone optimized for efficiency, not maximum segmentation accuracy. Heavier backbones (ResNet50, ResNet101) would be expected to perform significantly better.
- The data pipeline supports **50× augmentation** (`mode='augmented'`), which was not used in this experiment.

---

## Setup & Usage

### Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Installation

```bash
git clone https://github.com/Samar-git-hub/DeepLabV3-FracAtlas.git
cd DeepLabV3-FracAtlas

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### Dataset Setup

1. Download [FracAtlas](https://www.kaggle.com/datasets/ahmedmohammad/fracatlas) from Kaggle.
2. Extract into `data/FracAtlas/` so the structure is:

```
data/FracAtlas/
├── Annotations/
│   └── COCO JSON/
│       └── COCO_fracture_masks.json
├── Utilities/
│   └── Fracture Split/
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
├── images/
│   ├── Fractured/
│   └── Non_fractured/
└── dataset.csv
```

### Prepare Data

Generate resized images, masks, and split CSVs:

```bash
python src/data_pipeline/augment.py
```

This creates:
- `data/FracAtlas/processed/original_1024/` — resized train/valid/test splits
- `data/FracAtlas/masks/Fractured/` — original-resolution masks (for evaluation)
- `data/FracAtlas/Augmented_1024/` — 50× augmented training data (optional)

### Train

```bash
python scripts/train.py
```

Saves the best model checkpoint and training log to `experiments/mobilenetv3/Exp1_DeepLab_MobileNet_ImageNet/`.

### Evaluate

```bash
python scripts/run_evaluation.py
```

Computes IoU, Dice, HD95, Accuracy, AUC, GFLOPs, and parameter count on the test set.

---

## Project Structure

```
DeepLabV3-FracAtlas/
├── scripts/
│   ├── train.py              # Training loop with BCE + Dice loss
│   └── run_evaluation.py     # Test set evaluation entry point
├── src/
│   ├── data_pipeline/
│   │   ├── augment.py        # Data preparation: resize, augment, mask generation
│   │   └── dataset.py        # PyTorch Dataset class
│   ├── evaluation/
│   │   └── evaluate.py       # Full evaluation pipeline (IoU, Dice, HD95, AUC)
│   └── models/
│       └── deeplabv3_mobilenetv3.py  # Model definition
├── data/FracAtlas/            # Dataset (not tracked)
├── experiments/               # Saved models and logs
├── requirements.txt
└── README.md
```

---

## Evaluation Pipeline Details

The evaluation computes metrics at **original image resolution** to ensure clinical relevance:

1. Input image (resized + padded to 1024×1024) is fed through the model.
2. Sigmoid activation produces per-pixel fracture probabilities.
3. Predictions are **inverse-transformed**: padding is removed, then resized back to the original image dimensions.
4. Metrics are computed against the original-resolution ground truth masks.

| Metric | Description |
|---|---|
| **IoU** | Intersection over Union between predicted and ground truth fracture regions |
| **Dice** | Harmonic mean of precision and recall at the pixel level |
| **HD95** | 95th percentile Hausdorff Distance — measures boundary accuracy (in pixels) |
| **Accuracy** | Pixel-wise classification accuracy |
| **AUC** | Area under the ROC curve (micro-averaged across all pixels) |

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{turk2024impact,
  title={The impact of implementing backbone architectures on fracture segmentation in X-ray images},
  author={Turk, Salih and Bingol, Ozkan and Co{\c{s}}kun{\c{c}}ay, Ahmet and Aydin, Tolga},
  journal={Engineering Science and Technology, an International Journal},
  volume={59},
  pages={101883},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.jestch.2024.101883}
}
```

## License

MIT License
