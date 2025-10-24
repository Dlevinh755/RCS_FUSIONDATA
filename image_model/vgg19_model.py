
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import train_test_split


IMG_SIZE = 224
img_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.ConvertImageDtype(torch.float),
    transforms.Lambda(lambda x: x if x.ndim==3 else x.expand(3, *x.shape[1:])),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for p in vgg.parameters():
            p.requires_grad = False
        self.features = nn.Sequential(*list(vgg.features.children()), vgg.avgpool)
        self.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
    def forward(self, img_batch):  # (N,3,224,224)
        f = self.features(img_batch)
        f = torch.flatten(f, 1)
        f = self.classifier(f)  # (N, 4096)
        return f