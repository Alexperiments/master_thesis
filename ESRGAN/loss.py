import torch
import torch.nn as nn
from torchvision.models import vgg19
import config

# phi_5,4 5th conv layer before maxpooling but after activation

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)

class bright_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, input, target):
        input_brightness = torch.sum(input, dim=[2,3], keepdim=True)
        target_brightness = torch.sum(target, dim=[2,3], keepdim=True)
        return self.loss(input_brightness, target_brightness)
