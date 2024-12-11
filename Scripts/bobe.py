import torch
import torch.nn as nn
import torchvision.models as models

# Backbone based on ResNet-50
class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(Backbone, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove fully connected
    
    def forward(self, x):
        return self.feature_extractor(x)

# Keypoint Detection Head
class KeypointHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_keypoints=8):
        super(KeypointHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_anchors * num_keypoints * 2, kernel_size=1)  # 8 keypoints * 2 (x, y)
        )
    
    def forward(self, x):
        return self.head(x)

# Keypoint-only Model
class LegoKeypointDetector(nn.Module):
    def __init__(self, num_anchors=10, num_keypoints=8):
        super(LegoKeypointDetector, self).__init__()
        self.backbone = Backbone()
        in_channels = 2048  # Output of ResNet-50
        
        # Keypoint Detection Branch
        self.keypoint_head = KeypointHead(in_channels, num_anchors, num_keypoints)
    
    def forward(self, x):
        features = self.backbone(x)
        keypoint_predictions = self.keypoint_head(features)
        return keypoint_predictions




def compute_loss(keypoint_preds, keypoint_targets):

    mse_loss = nn.MSELoss()
    loss = mse_loss(keypoint_preds, keypoint_targets)
    return loss