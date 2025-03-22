# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SimpleModel(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1):
#         super(SimpleModel, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
        
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
        
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, padding=1)
        
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.conv3(x)  # No activation to keep raw output values
#         return x
    


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling for keypoint regression
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        #x = self.pool(x)
        return x