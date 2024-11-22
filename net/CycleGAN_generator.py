# -*- coding: utf-8 -*-
# @Author  : Refactored for CycleGAN
# @File    : Ushape_Trans.py

import torch.nn as nn
import torch
import torch.nn.functional as F


def weights_init_normal(m):
    """Initialize weights for the model."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResnetBlock(nn.Module):
    """Residual Block for the Generator."""
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    """CycleGAN Generator Network."""
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(CycleGANGenerator, self).__init__()
        # Initial convolutional block
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResnetBlock(in_features)]
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        # Output layer
        model += [
            nn.Conv2d(64, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class CycleGANDiscriminator(nn.Module):
    """CycleGAN Discriminator Network."""
    def __init__(self, input_nc):
        super(CycleGANDiscriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        in_features = 64
        out_features = in_features * 2
        for _ in range(3):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        model += [
            nn.Conv2d(in_features, 1, kernel_size=4, padding=1)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # Test the Generator and Discriminator
    input_nc = 3
    output_nc = 3
    img_size = 256

    gen = CycleGANGenerator(input_nc, output_nc)
    disc = CycleGANDiscriminator(input_nc)

    print(f"Generator architecture:\n{gen}")
    print(f"Discriminator architecture:\n{disc}")

    test_input = torch.randn(1, input_nc, img_size, img_size)
    gen_output = gen(test_input)
    disc_output = disc(gen_output)

    print(f"Generator output shape: {gen_output.shape}")
    print(f"Discriminator output shape: {disc_output.shape}")
