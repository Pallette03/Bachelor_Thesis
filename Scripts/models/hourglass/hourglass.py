import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return F.relu(x + residual)

class Hourglass(nn.Module):
    def __init__(self, depth, channels):
        super().__init__()
        self.depth = depth
        self.downsample = nn.MaxPool2d(2, stride=2)
        self.upsample = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        self.blocks = nn.ModuleList([ResidualBlock(channels, channels) for _ in range(2 * depth)])

    def forward(self, x):
        down_features = []
        sizes = []
        for i in range(self.depth):
            x = self.blocks[i](x)
            down_features.append(x)
            sizes.append(x.shape[2:])
            x = self.downsample(x)
        
        x = self.blocks[self.depth](x)
        
        for i in range(self.depth):
            x = self.upsample(x, sizes.pop()) + down_features.pop()
            x = self.blocks[self.depth + i](x)
        
        return x

class StackedHourglass(nn.Module):
    def __init__(self, num_stacks=2, num_channels=256):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, num_channels)
        )
        self.hourglasses = nn.ModuleList([Hourglass(depth=4, channels=num_channels) for _ in range(num_stacks)])
        self.output_layers = nn.ModuleList([nn.Conv2d(num_channels, 1, kernel_size=1) for _ in range(num_stacks)])

    def forward(self, x):
        original_size = x.shape[2:]
        x = self.preprocess(x)
        for i, (hg, out_layer) in enumerate(zip(self.hourglasses, self.output_layers)):
            x = hg(x)
            heatmap = out_layer(x)
        return F.interpolate(heatmap, size=original_size, mode='bilinear', align_corners=True)