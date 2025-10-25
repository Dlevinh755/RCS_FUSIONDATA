
import torch
import torch.nn as nn
from torchvision import models



class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        for p in vgg.parameters():
            p.requires_grad = False
        self.features = nn.Sequential(*list(vgg.features.children()), vgg.avgpool)
        self.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
    def forward(self, img_batch):  # (N,3,224,224)
        f = self.features(img_batch)
        f = torch.flatten(f, 1)
        f = self.classifier(f)  # (N, 4096)
        return f