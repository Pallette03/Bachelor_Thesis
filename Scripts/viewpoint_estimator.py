import torch
import torch.nn as nn
import torchvision.models as models

class ViewpointEstimator(nn.Module):
    def __init__(self, num_azimuth_bins, num_elevation_bins, num_rotation_bins):
        super(ViewpointEstimator, self).__init__()
        # Load pre-trained VGG-16 and remove the classification head
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features  # Extract convolutional feature layers

        # Define fully connected layers for classification
        self.fc_azimuth = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),  # Larger hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),              # Dropout for regularization
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),              # Dropout here as well
            nn.Linear(512, num_azimuth_bins)
        )
        self.fc_elevation = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_elevation_bins)
        )
        self.fc_rotation = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_rotation_bins)
        )

    def forward(self, x):
        # Extract features using VGG-16
        features = self.features(x)
        features = torch.flatten(features, start_dim=1)  # Flatten for fully connected layers

        # Pass through fully connected layers
        azimuth_out = self.fc_azimuth(features)
        elevation_out = self.fc_elevation(features)
        rotation_out = self.fc_rotation(features)

        return azimuth_out, elevation_out, rotation_out
