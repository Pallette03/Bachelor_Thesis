import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import kornia
from kornia.color import rgb_to_grayscale
from kornia.filters import GaussianBlur2d
from kornia.feature import gftt_response, harris_response


class feature_extractor(nn.Module):
    '''
        It loads both, the handcrafted and learnable blocks
    '''
    def __init__(self, in_channels=3):
        super(feature_extractor, self).__init__()

        self.hc_block = handcrafted_block()
        self.lb_block = learnable_block(in_channels=(in_channels+3))

    def forward(self, x):
        x_hc = self.hc_block(x)
        x_lb = self.lb_block(x_hc)
        return x_lb


def normalize(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize a tensor to the range [0, 1] per-sample.
    """
    B, C, H, W = tensor.shape
    flat = tensor.view(B, -1)
    min_val = flat.min(dim=1)[0].view(B, 1, 1, 1)
    max_val = flat.max(dim=1)[0].view(B, 1, 1, 1)
    return (tensor - min_val) / (max_val - min_val + eps)


class handcrafted_block(nn.Module):
    """
    GPU-accelerated handcrafted detectors: Shi-Tomasi (GFTT), Harris, and FAST.
    FAST will run on a GPU if supported, otherwise falls back to CPU via OpenCV.
    """
    def __init__(self,
                 harris_k: float = 0.04,
                 harris_thr: float = 0.01,
                 fast_thresh: int = 10,
                 fast_nonmax: bool = True):
        super().__init__()
        self.harris_k = harris_k
        self.harris_thr = harris_thr
        self.fast_thresh = fast_thresh
        self.fast_nonmax = fast_nonmax
        
        self.blur = GaussianBlur2d((3, 3), (1.0, 1.0))

    def forward(self, x):
        # x: (B, C, H, W) in [0,1]
        device = x.device

        # Convert to grayscale
        gray = rgb_to_grayscale(x)  # (B,1,H,W)
        B, _, H, W = gray.shape

        # Shi-Tomasi / Good Features to Track response
        shi = gftt_response(
            gray,
            grads_mode='sobel'
        )
        shi = self.blur(shi)
        shi = normalize(shi)

        # Harris response
        har = harris_response(
            gray,
            k=self.harris_k,
            grads_mode='sobel'
        )
        # Threshold and keep only strong corners
        har = torch.where(
            har > self.harris_thr * har.amax(dim=[2, 3], keepdim=True),
            har,
            torch.zeros_like(har)
        )
        har = self.blur(har)
        har = normalize(har)

        gray_cpu = (gray.squeeze(1) * 255).byte().cpu().numpy()  # (B, H, W)
        detector = cv2.FastFeatureDetector_create(
            threshold=self.fast_thresh,
            nonmaxSuppression=self.fast_nonmax
        )
        fast_maps = []
        for i in range(B):
            kps = detector.detect(gray_cpu[i], None)
            mask = np.zeros((H, W), dtype=np.float32)
            for kp in kps:
                y, x_pt = int(kp.pt[1]), int(kp.pt[0])
                if 0 <= x_pt < W and 0 <= y < H:
                    mask[y, x_pt] = 1.0
            heat = cv2.GaussianBlur(mask, (3, 3), 1.0)
            minv, maxv = heat.min(), heat.max()
            if maxv - minv > 1e-8:
                heat = (heat - minv) / (maxv - minv)
            fast_maps.append(torch.from_numpy(heat).unsqueeze(0))  # (1, H, W)
        # Stack to (B, 1, H, W)
        fast = torch.stack(fast_maps, dim=0).to(device)

        # Concatenate original image with heatmaps
        out = torch.cat([x, har, shi, fast], dim=1)
        return out

class learnable_block(nn.Module):
    '''
        It defines the learnable blocks within the Key.Net
    '''
    def __init__(self, in_channels=10):
        super(learnable_block, self).__init__()

        self.conv0 = conv_blck(in_channels)
        self.conv1 = conv_blck()
        self.conv2 = conv_blck()#out_channels=in_channels)

    def forward(self, x):
        residual = x
        x = self.conv2(self.conv1(self.conv0(x)))
        
        return x# + residual
        


def conv_blck(in_channels=8, out_channels=8, kernel_size=5,
              stride=1, padding=2, dilation=1):
    '''
    Default learnable convolutional block.
    '''
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


class NonMaxSuppression(torch.nn.Module):
    '''
        NonMaxSuppression class
    '''
    def __init__(self, thr=0.0, nms_size=5):
        nn.Module.__init__(self)
        padding = nms_size // 2
        self.max_filter = torch.nn.MaxPool2d(kernel_size=nms_size, stride=1, padding=padding)
        self.thr = thr

    def forward(self, scores):

        # local maxima
        maxima = (scores == self.max_filter(scores))

        # remove low peaks
        maxima *= (scores > self.thr)

        return maxima.nonzero().t()[2:4]