import torch.nn as nn
import torch.nn.functional as F
import torch

class KeypointDetector(nn.Module):
    def __init__(self, input_features=64 * 28 * 28):
        super(KeypointDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(input_features, 512)
        self.fc2 = nn.Linear(512, 2)  # Outputs per keypoint (x, y)

    def forward(self, x, max_keypoints):
        batch_size = x.size(0)

        # Feature extraction
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)

        # Flatten features
        x = x.view(batch_size, -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))

        # Predict (x, y) for each keypoint
        x = self.fc2(x)

        # Expand to match [Batch, MaxKeypoints, 2]
        x = x.repeat(1, max_keypoints).view(batch_size, max_keypoints, 2)
        return x

