# import torch
# import torch.nn as nn


# def weights_init_normal(m):
#     """Initialize weights for the model."""
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("BatchNorm2d") != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)


# class ResidualBlock(nn.Module):
#     """Residual Block for ResNetGenerator."""
#     def __init__(self, dim):
#         super(ResidualBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
#             nn.InstanceNorm2d(dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
#             nn.InstanceNorm2d(dim),
#         )

#     def forward(self, x):
#         return x + self.block(x)


# class ResnetGenerator(nn.Module):
#     """ResNet Generator (standard CycleGAN generator)."""
#     def __init__(self, input_nc, output_nc, n_residual_blocks=9):
#         super(ResnetGenerator, self).__init__()
#         # Initial convolution block
#         model = [
#             nn.Conv2d(input_nc, 64, kernel_size=7, padding=3, bias=False),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True),
#         ]
#         # Downsampling
#         in_features = 64
#         out_features = in_features * 2
#         for _ in range(2):
#             model += [
#                 nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.InstanceNorm2d(out_features),
#                 nn.ReLU(inplace=True),
#             ]
#             in_features = out_features
#             out_features = in_features * 2
#         # Residual blocks
#         for _ in range(n_residual_blocks):
#             model += [ResidualBlock(in_features)]
#         # Upsampling
#         out_features = in_features // 2
#         for _ in range(2):
#             model += [
#                 nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
#                 nn.InstanceNorm2d(out_features),
#                 nn.ReLU(inplace=True),
#             ]
#             in_features = out_features
#             out_features = in_features // 2
#         # Output layer
#         model += [
#             nn.Conv2d(64, output_nc, kernel_size=7, padding=3),
#             nn.Tanh(),
#         ]
#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         return self.model(x)


# class EnhancedCycleGANGenerator(nn.Module):
#     """Enhanced CycleGAN Generator with Residual Blocks and Attention."""
#     def __init__(self, input_nc, output_nc, n_residual_blocks=9):
#         super(EnhancedCycleGANGenerator, self).__init__()
#         # Initial convolutional block
#         model = [
#             nn.Conv2d(input_nc, 64, kernel_size=7, padding=3, bias=False),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True),
#         ]
#         # Downsampling
#         in_features = 64
#         out_features = in_features * 2
#         for _ in range(2):
#             model += [
#                 nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.InstanceNorm2d(out_features),
#                 nn.ReLU(inplace=True),
#             ]
#             in_features = out_features
#             out_features = in_features * 2
#         # Residual blocks
#         for _ in range(n_residual_blocks):
#             model += [ResidualBlock(in_features)]
#         # Upsampling
#         out_features = in_features // 2
#         for _ in range(2):
#             model += [
#                 nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
#                 nn.InstanceNorm2d(out_features),
#                 nn.ReLU(inplace=True),
#             ]
#             in_features = out_features
#             out_features = in_features // 2
#         # Output layer
#         model += [
#             nn.Conv2d(64, output_nc, kernel_size=7, padding=3),
#             nn.Tanh(),
#         ]
#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         return self.model(x)


# class CycleGANDiscriminator(nn.Module):
#     """CycleGAN Patch Discriminator."""
#     def __init__(self, input_nc=3):
#         super(CycleGANDiscriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512, 1, kernel_size=4, padding=1),
#         )

#     def forward(self, x):
#         return self.model(x)


# class CustomResnetGenerator(nn.Module):
#     """Custom ResNet Generator to match the weight keys."""
#     def __init__(self, input_nc, output_nc):
#         super(CustomResnetGenerator, self).__init__()

#         self.model = nn.Sequential(
#             nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3),
#             nn.ReLU(inplace=True),
#             self.conv_block(64, 128),
#             self.conv_block(128, 256),
#             self.conv_block(256, 512),
#             self.upconv_block(512, 256),
#             self.upconv_block(256, 128),
#             self.upconv_block(128, 64),
#             nn.Conv2d(64, output_nc, kernel_size=7, stride=1, padding=3),
#             nn.Tanh(),
#         )

#     def conv_block(self, in_channels, out_channels):
#         """Convolutional block with Instance Norm."""
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
#             nn.InstanceNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def upconv_block(self, in_channels, out_channels):
#         """Upsampling block with Instance Norm."""
#         return nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
#             ),
#             nn.InstanceNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.model(x)


# if __name__ == "__main__":
#     # Test the ResnetGenerator and EnhancedCycleGANGenerator
#     input_nc = 3
#     output_nc = 3
#     img_size = 256

#     # Instantiate models
#     resnet_gen = ResnetGenerator(input_nc, output_nc)
#     cyclegan_gen = EnhancedCycleGANGenerator(input_nc, output_nc)
#     disc = CycleGANDiscriminator(input_nc)

#     print(f"ResNet Generator:\n{resnet_gen}")
#     print(f"Enhanced CycleGAN Generator:\n{cyclegan_gen}")
#     print(f"Discriminator:\n{disc}")

#     # Test input
#     test_input = torch.randn(1, input_nc, img_size, img_size)
#     resnet_output = resnet_gen(test_input)
#     cyclegan_output = cyclegan_gen(test_input)
#     disc_output = disc(test_input)

#     print(f"ResNet Generator output shape: {resnet_output.shape}")
#     print(f"CycleGAN Generator output shape: {cyclegan_output.shape}")
#     print(f"Discriminator output shape: {disc_output.shape}")



import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """Define a ResNet block."""
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            conv_block += [nn.ZeroPad2d(1)]
        else:
            raise NotImplementedError(f"Padding type {padding_type} is not implemented")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            conv_block += [nn.ZeroPad2d(1)]

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """ResNet Generator from the CycleGAN repository."""
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        use_bias = (type(norm_layer) == nn.InstanceNorm2d)

        # Initial Convolution Block
        model = [
            nn.ReflectionPad2d(3),  # Padding before the first Conv2d layer
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        # Residual Blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type='reflect', norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult, int(ngf * mult / 2),
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(inplace=True),
            ]

        # Output Layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)



class CycleGANDiscriminator(nn.Module):
    """CycleGAN PatchGAN Discriminator."""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(CycleGANDiscriminator, self).__init__()
        use_bias = (type(norm_layer) == nn.InstanceNorm2d)

        # Initial Layer
        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Hidden Layers
        for i in range(1, n_layers):
            mult = 2 ** (i - 1)
            model += [
                nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * mult * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Final Layer
        mult = 2 ** (n_layers - 1)
        model += [
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


if __name__ == "__main__":
    # Test the ResnetGenerator
    input_nc = 3
    output_nc = 3
    img_size = 256

    # Instantiate models
    resnet_gen = ResnetGenerator(input_nc, output_nc)
    discriminator = CycleGANDiscriminator(input_nc)

    # Test input
    test_input = torch.randn(1, input_nc, img_size, img_size)
    resnet_output = resnet_gen(test_input)
    discriminator_output = discriminator(test_input)

    print(f"ResNet Generator output shape: {resnet_output.shape}")
    print(f"CycleGAN Discriminator output shape: {discriminator_output.shape}")