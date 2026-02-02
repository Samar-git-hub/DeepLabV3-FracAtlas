import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models import MobileNet_V3_Large_Weights

def get_deeplab_model(device, model_weights='ImageNet'):

    print("Loading DeepLabV3 + MobileNetV3 Large")

    if model_weights == "ImageNet":
        weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        weights = None

    elif model_weights == "COCO":
        weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        weights_backbone = None
        
    else:
        weights = None
        weights_backbone = None
    
    model = deeplabv3_mobilenet_v3_large(
        weights=weights,
        weights_backbone=weights_backbone,
        aux_loss=True
    )

    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1,1), stride=(1,1))

    aux_in_channels = model.aux_classifier[4].in_channels
    model.aux_classifier[4] = nn.Conv2d(aux_in_channels, 1, kernel_size=(1,1), stride=(1,1))

    return model.to(device)