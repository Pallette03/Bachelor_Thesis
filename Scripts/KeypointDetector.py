import torch
import torch.nn as nn
import torch.nn.functional as F

class KeypointDetector(nn.Module):
    def __init__(self, num_bricks=10):
        super(KeypointDetector, self).__init__()
        self.num_bricks = num_bricks
        self.num_corners = 8
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_bricks * self.num_corners * 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x.view(-1, self.num_bricks, self.num_corners, 2)
