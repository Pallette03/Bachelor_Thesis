import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import kornia
from scipy.ndimage import gaussian_filter


class feature_extractor(nn.Module):
    '''
        It loads both, the handcrafted and learnable blocks
    '''
    def __init__(self, in_channels=3):
        super(feature_extractor, self).__init__()

        self.hc_block = handcrafted_block()
        self.lb_block = learnable_block(in_channels=(in_channels+2))

    def forward(self, x):
        x_hc = self.hc_block(x)
        x_lb = self.lb_block(x_hc)
        return x_lb


class handcrafted_block(nn.Module):
    '''
        It defines the handcrafted filters within the Key.Net handcrafted block
    '''
    def __init__(self):
        super(handcrafted_block, self).__init__()

    def add_fast_keypoint_heatmap(self, tensor_images: torch.Tensor, sigma: float = 2.0):
        """
        Takes a batch of PyTorch tensor images, extracts FAST keypoints using OpenCV,
        creates a heatmap, applies Gaussian blur, and appends it as a new channel.
        
        Args:
            tensor_images (torch.Tensor): Input batch of images (B, C, H, W) in range [0,1].
            sigma (float): Standard deviation for Gaussian blur.
        
        Returns:
            torch.Tensor: Modified tensor with an extra channel containing the heatmap.
        """
        if tensor_images.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions (B, C, H, W)")
        
        device = tensor_images.device
        batch_size, _, height, width = tensor_images.shape
        heatmaps = []
        
        for img in tensor_images:
            # Convert tensor to OpenCV format (H, W, C) and then grayscale
            np_image = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY) if np_image.shape[-1] == 3 else np_image
            
            # Use FAST feature detector
            fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=False)
            keypoints = fast.detect(gray, None)
            
            # Create heatmap
            heatmap = np.zeros((height, width), dtype=np.float32)
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= x < width and 0 <= y < height:
                    heatmap[y, x] = 1.0
            
            # Apply Gaussian blur
            heatmap = gaussian_filter(heatmap, sigma=sigma)
            
            # Normalize heatmap to [0,1]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Convert to PyTorch tensor
            heatmaps.append(torch.from_numpy(heatmap).unsqueeze(0))  # (1, H, W)
        
        heatmaps_tensor = torch.stack(heatmaps, dim=0).to(device)  # (B, 1, H, W)
        #output_tensor = torch.cat([tensor_images, heatmaps_tensor], dim=1)  # (B, C+1, H, W)
        
        return heatmaps_tensor
    
    def harris_laplace_detector_batch(self, images, num_octaves=6, initial_sigma=1.4, k=0.04, threshold=0.01):
        """
        Apply the Harris-Laplace corner detector to a batch of images.

        Parameters:
        - images: Batch of images as a PyTorch tensor of shape (N, C, H, W).
        - num_octaves: Number of octaves for Laplacian computation.
        - initial_sigma: Initial sigma for Gaussian blur.
        - k: Harris detector free parameter.
        - threshold: Threshold for Harris response.
        - output_dir: Directory to save the output images.

        Returns:
        - corners_batch: Batch of corner maps as a NumPy array of shape (N, 1, H, W).
        """
        batch_size, channels, height, width = images.shape
        corners_batch = np.zeros((batch_size, height, width), dtype=np.float32)

        device = images.device
        images = images.cpu().numpy()

        for batch_idx in range(batch_size):
            image = images[batch_idx]
            combined_corners = np.zeros((height, width), dtype=np.float32)

            for channel_idx in range(channels):
                gray = cv2.normalize(image[channel_idx], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                harris_corners = cv2.cornerHarris(gray, 2, 3, k)
                harris_corners = cv2.dilate(harris_corners, None)
                harris_corners = harris_corners > threshold * harris_corners.max()

                for octave in range(num_octaves):
                    sigma = initial_sigma * (2 ** octave)
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3, scale=sigma)
                    laplacian = cv2.GaussianBlur(laplacian, (0, 0), sigma)
                    laplacian = np.abs(laplacian)

                    combined_response = harris_corners * laplacian
                    combined_corners = np.maximum(combined_corners, combined_response)

            corners_batch[batch_idx] = combined_corners
            #cv2.imwrite(os.path.join(output_dir, f'harris_laplace_corners_{batch_idx}.png'), (combined_corners * 255).astype(np.uint8))

        # Convert to tensor and return
        corners_batch = torch.from_numpy(corners_batch)
        # add channel dimension
        corners_batch = corners_batch.unsqueeze(1)

        return corners_batch.to(device)
    
    def canny_into_harris(self, input_tensor):
        B, C, H, W = input_tensor.shape
        device = input_tensor.device
        all_responses = torch.zeros(B, 1, H, W, device=device)

        for i in range(B):
            image_tensor = input_tensor[i].to(torch.float32).cpu()

            if C == 3:
                # Split RGB channels and convert to grayscale
                r, g, b = image_tensor[0], image_tensor[1], image_tensor[2]
                gray = 0.299 * r + 0.587 * g + 0.114 * b
            else:
                gray = image_tensor.squeeze(0)

            if r is not None:
                r_canny = cv2.Canny((r.numpy() * 255).astype(np.uint8), 100, 200)
                g_canny = cv2.Canny((g.numpy() * 255).astype(np.uint8), 100, 200)
                b_canny = cv2.Canny((b.numpy() * 255).astype(np.uint8), 100, 200)
                gray_canny = cv2.Canny((gray.numpy() * 255).astype(np.uint8), 100, 200)

                # Use good features to track to find corners
                r_keypoints = cv2.goodFeaturesToTrack(r_canny, 100, 0.01, 10)
                g_keypoints = cv2.goodFeaturesToTrack(g_canny, 100, 0.01, 10)
                b_keypoints = cv2.goodFeaturesToTrack(b_canny, 100, 0.01, 10)
                gray_keypoints = cv2.goodFeaturesToTrack(gray_canny, 100, 0.01, 10)

                # Create a heatmap for all keypoints
                heatmap = np.zeros((H, W), dtype=np.float32)

                # Add keypoints from each channel
                for keypoints in [r_keypoints, g_keypoints, b_keypoints, gray_keypoints]:
                    if keypoints is not None:
                        for keypoint in keypoints:
                            x, y = keypoint.ravel()
                            heatmap[int(y), int(x)] += 1

                # Normalize the heatmap
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

                # Convert heatmap to tensor and add to responses
                all_keypoints = torch.from_numpy(heatmap).unsqueeze(0).to(device)
            else:
                gray_canny = cv2.Canny((gray.numpy() * 255).astype(np.uint8), 100, 200)
                gray_keypoints = cv2.goodFeaturesToTrack(gray_canny, 100, 0.01, 10)
                
                heatmap = np.zeros((H, W), dtype=np.float32)
                if gray_keypoints is not None:
                    for keypoint in gray_keypoints:
                        x, y = keypoint.ravel()
                        heatmap[int(y), int(x)] += 1
                    
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                all_keypoints = torch.from_numpy(heatmap).unsqueeze(0).to(device)


            # Apply Gaussian blur to the heatmap
            blurred_heatmap = cv2.GaussianBlur(heatmap, (5, 5), sigmaX=1.0, sigmaY=1.0)

            # Normalize the blurred heatmap
            blurred_heatmap = (blurred_heatmap - blurred_heatmap.min()) / (blurred_heatmap.max() - blurred_heatmap.min() + 1e-8)

            # Convert blurred heatmap to tensor and add to responses
            all_keypoints = torch.from_numpy(blurred_heatmap).unsqueeze(0).to(device)

            all_responses[i] = all_keypoints

        return all_responses.to(device)

    def good_features_to_track(self, input_tensor, max_corners=100, quality_level=0.1, min_distance=5):
        B, C, H, W = input_tensor.shape
        device = input_tensor.device
        all_responses = torch.zeros(B, 1, H, W, device='cpu')

        for i in range(B):
            image_tensor = input_tensor[i].to(torch.float32).cpu()

            if C == 3:
                # Split RGB channels and convert to grayscale
                r, g, b = image_tensor[0], image_tensor[1], image_tensor[2]
                gray = 0.299 * r + 0.587 * g + 0.114 * b
            else:
                gray = image_tensor.squeeze(0)

            if r is not None:
                r_features = cv2.goodFeaturesToTrack((r.numpy() * 255).astype(np.uint8), max_corners, quality_level, min_distance)
                g_features = cv2.goodFeaturesToTrack((g.numpy() * 255).astype(np.uint8), max_corners, quality_level, min_distance)
                b_features = cv2.goodFeaturesToTrack((b.numpy() * 255).astype(np.uint8), max_corners, quality_level, min_distance)
                gray_features = cv2.goodFeaturesToTrack((gray.numpy() * 255).astype(np.uint8), max_corners, quality_level, min_distance)

                # Create a heatmap for all keypoints
                heatmap = np.zeros((H, W), dtype=np.float32)

                # Add keypoints from each channel
                for features in [r_features, g_features, b_features, gray_features]:
                    if features is not None:
                        for feature in features:
                            x, y = feature.ravel()
                            heatmap[int(y), int(x)] += 1

                # Normalize the heatmap
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

                # add gaussian blur
                blurred_heatmap = cv2.GaussianBlur(heatmap, (5, 5), sigmaX=1.0, sigmaY=1.0)

                # Normalize the blurred heatmap
                blurred_heatmap = (blurred_heatmap - blurred_heatmap.min()) / (blurred_heatmap.max() - blurred_heatmap.min() + 1e-8)

                # Convert blurred heatmap to tensor and add to responses
                all_keypoints = torch.from_numpy(blurred_heatmap).unsqueeze(0).to(device)
            else:
                gray_features = cv2.goodFeaturesToTrack((gray.numpy() * 255).astype(np.uint8), max_corners, quality_level, min_distance)
                
                heatmap = np.zeros((H, W), dtype=np.float32)
                if gray_features is not None:
                    for feature in gray_features:
                        x, y = feature.ravel()
                        heatmap[int(y), int(x)] += 1
                    
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                blurred_heatmap = cv2.GaussianBlur(heatmap, (5, 5), sigmaX=1.0, sigmaY=1.0)
                blurred_heatmap = (blurred_heatmap - blurred_heatmap.min()) / (blurred_heatmap.max() - blurred_heatmap.min() + 1e-8)

                all_keypoints = torch.from_numpy(blurred_heatmap).unsqueeze(0).to(device)

            all_responses[i] = all_keypoints

        return all_responses.to(device)

    def add_fast_keypoint_heatmap(self, tensor_images: torch.Tensor, sigma: float = 2.0):
        """
        Takes a batch of PyTorch tensor images, extracts FAST keypoints using OpenCV,
        creates a heatmap, applies Gaussian blur, and appends it as a new channel.
        
        Args:
            tensor_images (torch.Tensor): Input batch of images (B, C, H, W) in range [0,1].
            sigma (float): Standard deviation for Gaussian blur.
        
        Returns:
            torch.Tensor: Modified tensor with an extra channel containing the heatmap.
        """
        if tensor_images.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions (B, C, H, W)")
        
        device = tensor_images.device
        batch_size, _, height, width = tensor_images.shape
        heatmaps = []
        
        for img in tensor_images:
            # Convert tensor to OpenCV format (H, W, C) and then grayscale
            np_image = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY) if np_image.shape[-1] == 3 else np_image
            
            # Use FAST feature detector
            fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=False)
            keypoints = fast.detect(gray, None)
            
            # Create heatmap
            heatmap = np.zeros((height, width), dtype=np.float32)
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= x < width and 0 <= y < height:
                    heatmap[y, x] = 1.0
            
            # Apply Gaussian blur
            heatmap = gaussian_filter(heatmap, sigma=sigma)
            
            # Normalize heatmap to [0,1]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Convert to PyTorch tensor
            heatmaps.append(torch.from_numpy(heatmap).unsqueeze(0))  # (1, H, W)
        
        heatmaps_tensor = torch.stack(heatmaps, dim=0).to(device)  # (B, 1, H, W)
        #output_tensor = torch.cat([tensor_images, heatmaps_tensor], dim=1)  # (B, C+1, H, W)
        
        return heatmaps_tensor
    
    def harris_detector_batch(self, images, k=0.04, threshold=0.01):
        batch_size, channels, height, width = images.shape
        corners_batch = np.zeros((batch_size, height, width), dtype=np.float32)

        device = images.device
        images = images.cpu().numpy()

        for batch_idx in range(batch_size):
            image = images[batch_idx]
            combined_corners = np.zeros((height, width), dtype=np.float32)

            for channel_idx in range(channels):
                gray = cv2.normalize(image[channel_idx], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                harris_corners = cv2.cornerHarris(gray, 2, 3, k)
                harris_corners = cv2.dilate(harris_corners, None)
                harris_corners = harris_corners > threshold * harris_corners.max()

                combined_corners = np.maximum(combined_corners, harris_corners)

            blurred_corners = cv2.GaussianBlur(combined_corners, (5, 5), sigmaX=1.0, sigmaY=1.0)
            blurred_corners = (blurred_corners - blurred_corners.min()) / (blurred_corners.max() - blurred_corners.min() + 1e-8)

            corners_batch[batch_idx] = blurred_corners
            #cv2.imwrite(os.path.join(output_dir, f'harris_laplace_corners_{batch_idx}.png'), (combined_corners * 255).astype(np.uint8))

        # Convert to tensor and return
        corners_batch = torch.from_numpy(corners_batch)
        # add channel dimension
        corners_batch = corners_batch.unsqueeze(1)

        return corners_batch.to(device)



    def forward(self, x):

        fast_keypoint_heatmap = self.add_fast_keypoint_heatmap(x)
        #harris_laplace_heatmap = self.harris_laplace_detector_batch(x)
        #canny_into_harris_heatmap = self.canny_into_harris(x)
        #good_features_heatmap = self.good_features_to_track(x)
        harris_heatmap = self.harris_detector_batch(x)

        x = torch.cat((x, fast_keypoint_heatmap, harris_heatmap), dim=1)

        return x


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