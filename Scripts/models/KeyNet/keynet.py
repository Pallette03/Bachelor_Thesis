import torch
import torch.nn as nn
import torch.nn.functional as F
from models.KeyNet.keynet_modules import feature_extractor
from models.KeyNet.kornia_utils import custom_pyrdown

class KeyNet(nn.Module):
    '''
    Key.Net model definition
    '''
    def __init__(self, num_filters, num_levels, kernel_size, in_channels=3):
        super(KeyNet, self).__init__()

        self.num_filters = num_filters
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.feature_extractor = feature_extractor(in_channels=in_channels)
        self.last_conv = nn.Sequential(nn.Conv2d(in_channels=self.num_filters*self.num_levels,
                                                 out_channels=1, kernel_size=self.kernel_size, padding=padding))

    def forward(self, x):
        """
        x - input image
        """
        shape_im = x.shape
        for i in range(self.num_levels):
            if i == 0:
                feats = self.feature_extractor(x)
            else:
                x = custom_pyrdown(x, factor=1.2)
                feats_i = self.feature_extractor(x)
                feats_i = F.interpolate(feats_i, size=(shape_im[2], shape_im[3]), mode='bilinear')
                feats = torch.cat([feats, feats_i], dim=1)

        scores = self.last_conv(feats)
        return scores