import torch
import torch.nn as nn
from torchvision import models

adversarial_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

class VGGLoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()
        # indices corresponding to relu1_2, relu2_2, relu3_4, relu4_4-ish
        self.layers = [2,7,16,25]
    def to_3ch(self, x):
        # x in [-1,1] -> [0,1] -> repeat to 3 channels
        x = (x + 1.0) * 0.5
        return x.repeat(1,3,1,1)
    def forward(self, x, y):
        x3 = self.to_3ch(x)
        y3 = self.to_3ch(y)
        loss = 0.0
        xi, yi = x3, y3
        for i, layer in enumerate(self.vgg):
            xi = layer(xi); yi = layer(yi)
            if i in self.layers:
                loss += self.criterion(xi, yi)
        return loss
