import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super(UnetBlockDown, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, padding=1, padding_mode='reflect')
        self.batch_norm_1 = nn.InstanceNorm2d(out_channels)
        self.batch_norm_2 = nn.InstanceNorm2d(out_channels)

        self.activation = activation
        if activation is None:
            self.activation = nn.LeakyReLU(0.2)

        # initialize parameters
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, x):
        x = self.activation(self.batch_norm_1(self.conv1(x)))
        x = self.activation(self.batch_norm_2(self.conv2(x)))
        skip = x
        x = self.max_pool(x)
        return x, skip


class UnetBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None, up=True):
        super(UnetBlockUp, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(out_channels, out_channels, bias=False, kernel_size=3, padding=1, padding_mode='reflect')
        self.batch_norm_1 = nn.InstanceNorm2d(out_channels)
        self.batch_norm_2 = nn.InstanceNorm2d(out_channels)

        self.up = up
        if up:
            self.up_conv = nn.ConvTranspose2d(out_channels, out_channels // 2, bias=False, kernel_size=2, stride=2)
        self.activation = activation
        if activation is None:
            self.activation = nn.LeakyReLU(0.2)

        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, x, skip):
        x = torch.cat([skip, x], dim=1)
        x = self.activation(self.batch_norm_1(self.conv1(x)))
        x = self.activation(self.batch_norm_2(self.conv2(x)))

        if self.up:
            x = self.up_conv(x)

        return x


class Generator(nn.Module):
    def __init__(self, in_list, out_list, img_size=None, num_channels=2):
        super(Generator, self).__init__()

        self.img_size = img_size
        down_pairs = [(in_list[i-1], in_list[i]) for i in range(1, len(in_list))]
        up_pairs = [(out_list[i-1], out_list[i]) for i in range(1, len(out_list))]

        self.down_blocks = nn.ModuleList()
        for in_c, out_c in down_pairs:
            self.down_blocks.append(UnetBlockDown(in_c, out_c))

        self.middle_part = nn.Sequential(
            nn.Conv2d(in_list[-1], in_list[-1]*2, bias=False, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_list[-1] * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_list[-1]*2, in_list[-1], bias=False, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_list[-1]),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_list[-1], in_list[-1], bias=False, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2)
        )
        for m in self.middle_part.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.02)

        self.up_blocks = nn.ModuleList()
        for in_c, out_c in up_pairs[:-1]:
            self.up_blocks.append(UnetBlockUp(in_c, out_c))
        self.up_blocks.append((UnetBlockUp(*up_pairs[-1], up=False)))

        self.final_conv = nn.Conv2d(out_list[-1], num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        skips = []
        for i, down_block in enumerate(self.down_blocks):
            x, skip = down_block(x)
            skips.append(skip)

        skips = skips[::-1]
        x = self.middle_part(x)

        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, skips[i])

        x = self.final_conv(x)

        return nn.Tanh()(x)




