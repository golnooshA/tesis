import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.layers(input)
        target_features = self.layers(target)
        return torch.nn.functional.l1_loss(input_features, target_features)

