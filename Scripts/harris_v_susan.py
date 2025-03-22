import os
import scipy
import torch.nn.functional as F
import cv2
import numpy as np
import torch
import torchvision
from scipy.ndimage import gaussian_filter

from LegoKeypointDataset import LegoKeypointDataset


annotations_folder = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'test', 'annotations')
img_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'test', 'images')
model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', 'dynamic_corner_detector_epoch.pth')

global_image_size = (1000, 1000)

def collate_fn(batch):
    images = [item["image"] for item in batch]
    corners_list = [item["norm_corners"] for item in batch]
    max_corner_amount = max([norm_corners.shape[0] for norm_corners in corners_list])

    # Pad the corners
    for i in range(len(corners_list)):
        corners = corners_list[i]
        pad_amount = max_corner_amount - corners.shape[0]
        pad = np.zeros((pad_amount, 2))
        corners_list[i] = np.concatenate((corners, pad), axis=0)
        


    images = torch.stack(images)
    corners_list = torch.stack([torch.tensor(corners, dtype=torch.float32) for corners in corners_list])

    return {"image": images, "norm_corners": corners_list}


def harris_corner_rgb(image, block_size=2, ksize=3, k=0.04, output_dir='corner_output'):
    """
    Compute Harris corner detection on each RGB channel and combine results.

    Parameters:
    - image: Input RGB image (NumPy array).
    - block_size: Neighborhood size for Harris detector.
    - ksize: Aperture parameter for Sobel derivative.
    - k: Harris detector free parameter.

    Returns:
    - corner_response: Combined corner response map.
    """
    # Split image into R, G, B channels
    b, g, r = cv2.split(image)

    # Compute Harris response for each channel
    harris_r = cv2.cornerHarris(r, block_size, ksize, k)
    harris_g = cv2.cornerHarris(g, block_size, ksize, k)
    harris_b = cv2.cornerHarris(b, block_size, ksize, k)

    # Combine responses 
    corner_response = harris_r + harris_g + harris_b

    # Normalize response for visualization
    corner_response = cv2.normalize(corner_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, 'harris_corners.png'), corner_response)

    return corner_response


def shi_tomasi_corner_detector(image, max_corners=100, quality_level=0.01, min_distance=10):
    """
    Apply the Shi-Tomasi corner detector to each channel of the image.
    """
    corners = np.zeros_like(image)
    for i in range(image.shape[0]):
        gray = cv2.normalize(image[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        detected_corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
        if detected_corners is not None:
            for corner in detected_corners:
                x, y = corner.ravel()
                corners[i, int(y), int(x)] = 1
    return corners

def harris_laplace_detector_batch(images, num_octaves=6, initial_sigma=1.4, k=0.04, threshold=0.01, output_dir='corner_output'):
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
    - corners_batch: Batch of corner maps as a NumPy array of shape (N, H, W).
    """
    os.makedirs(output_dir, exist_ok=True)
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

    # save the last image as well as its input
    cv2.imwrite(os.path.join(output_dir, 'input_image_bat.png'), (images[-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, f'harris_laplace_corners_{batch_idx}.png'), (corners_batch[-1] * 255).astype(np.uint8))

    # Convert to tensor and return
    corners_batch = torch.from_numpy(corners_batch)
    # add channel dimension
    corners_batch = corners_batch.unsqueeze(1)

    return corners_batch.to(device)

def susan_corner_detector(input_tensor, radius=3, threshold=27, corner_threshold=0.5, output_dir='corner_output'):
    """
    SUSAN Corner Detector implementation.
    
    Parameters:
    - image: Grayscale input image (NumPy array).
    - radius: Radius of the circular mask.
    - threshold: Intensity difference threshold for similarity.
    - corner_threshold: Sensitivity for detecting corners (lower = more sensitive).
    
    Returns:
    - corner_map: Binary image with corners marked as 255.
    """
    B, C, H, W = input_tensor.shape
    device = input_tensor.device
    all_responses = torch.zeros(B, 1, H, W, device=device)

    for i in range(B):
        image_tensor = input_tensor[i].to(torch.float32).cpu()

        if C == 3:
            # Split RGB channels and convert to grayscale
            r, g, b = image_tensor[0].numpy(), image_tensor[1].numpy(), image_tensor[2].numpy()
            gray = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            gray = image_tensor.squeeze(0).numpy()

        corner_map = np.zeros((H, W), dtype=np.uint8)

        # Create a circular mask (USAN region)
        mask = []
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                if x**2 + y**2 <= radius**2:  # Circle equation
                    mask.append((y, x))

        # Apply the SUSAN operator
        for y in range(radius, H - radius):
            for x in range(radius, W - radius):
                nucleus = gray[y, x]
                similar_pixels = 0
                
                for dy, dx in mask:
                    if abs(int(gray[y + dy, x + dx]) - int(nucleus)) < threshold:
                        similar_pixels += 1  # Count similar pixels

                # Compute SUSAN response
                s = similar_pixels / len(mask)
                response = 1 - s  # High response means corner
                
                # Mark corners based on the response
                if response > corner_threshold:
                    corner_map[y, x] = 255

    cv2.imwrite(os.path.join(output_dir, 'susan_corners.png'), corner_map)
    return corner_map


def fast_corner(input_image, output_dir='corner_output'):

    # Convert from numpy to OpenCV format
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply FAST Corner Detector
    fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=False)
    keypoints = fast.detect(gray, None)
    #print(f"Number of FAST corners: {len(keypoints)}")
    #Create a heatmap
    heatmap = np.zeros_like(gray)
    for keypoint in keypoints:
        x, y = keypoint.pt
        heatmap[int(y), int(x)] = 1
    

    cv2.imwrite(os.path.join(output_dir, 'fast_corners.png'), heatmap)
    return heatmap, keypoints

def gaussian_kernel(size: int, sigma: float, device='cuda'):
    """Creates a 2D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords[:, None] ** 2 + coords[None, :] ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def harris_response(image, k=0.04):
    """Computes the Harris corner response on a grayscale image."""
    # Sobel filters for gradient computation
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    
    Ix = F.conv2d(image, sobel_x, padding=1)
    Iy = F.conv2d(image, sobel_y, padding=1)
    
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # Apply Gaussian smoothing
    gauss = gaussian_kernel(5, 1.0, device=image.device).view(1, 1, 5, 5)
    Sxx = F.conv2d(Ixx, gauss, padding=2)
    Syy = F.conv2d(Iyy, gauss, padding=2)
    Sxy = F.conv2d(Ixy, gauss, padding=2)
    
    # Harris response: det(M) - k * (trace(M))^2
    detM = Sxx * Syy - Sxy * Sxy
    traceM = Sxx + Syy
    R = detM - k * traceM ** 2
    return R

def laplacian_response(image):
    """Computes the Laplacian-of-Gaussian (LoG) response."""
    log_filter = torch.tensor([[0, 0, -1, 0, 0],
                               [0, -1, -2, -1, 0],
                               [-1, -2, 16, -2, -1],
                               [0, -1, -2, -1, 0],
                               [0, 0, -1, 0, 0]], dtype=torch.float32, device=image.device).view(1, 1, 5, 5)
    return F.conv2d(image, log_filter, padding=2)

def harris_laplace_detector_gpu(input_tensor, threshold=0.3, output_dir='corner_output'):
    """Detects keypoints using the Harris-Laplace detector."""
    B, C, H, W = input_tensor.shape
    
    all_responses = torch.zeros(B, 1, H, W, device=input_tensor.device)
    for i in range(B):
        image_tensor = input_tensor[i].to(torch.float32)

        if C == 3:
            image_tensor = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
            image_tensor = image_tensor.unsqueeze(0)
    
        harris = harris_response(image_tensor)
        laplacian = laplacian_response(image_tensor)

        # Normalize responses
        harris = (harris - harris.min()) / (harris.max() - harris.min())
        laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())
        
        # Find points where both Harris and Laplacian are high
        response = harris * laplacian
        #keypoints = (response > threshold).nonzero(as_tuple=False)[:, 2:]
        all_responses[i] = response


    
    cv2.imwrite(os.path.join(output_dir, 'input_image_bat.png'), (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


    cv2.imwrite(os.path.join(output_dir, 'harris_laplace_corners_gpu.png'), (all_responses[-1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    
    return all_responses
    
def canny_into_harris(input_tensor, output_dir='corner_output'):
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
            r_canny = cv2.Sobel((r.numpy() * 255).astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
            g_canny = cv2.Sobel((g.numpy() * 255).astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
            b_canny = cv2.Sobel((b.numpy() * 255).astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
            gray_canny = cv2.Sobel((gray.numpy() * 255).astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)

            # Use good features to track to find corners
            r_keypoints = cv2.goodFeaturesToTrack(r_canny.astype(np.uint8), 100, 0.01, 10)
            g_keypoints = cv2.goodFeaturesToTrack(g_canny.astype(np.uint8), 100, 0.01, 10)
            b_keypoints = cv2.goodFeaturesToTrack(b_canny.astype(np.uint8), 100, 0.01, 10)
            gray_keypoints = cv2.goodFeaturesToTrack(gray_canny.astype(np.uint8), 100, 0.01, 10)

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
            gray_canny = cv2.Sobel((gray.numpy() * 255).astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
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

    cv2.imwrite(os.path.join(output_dir, 'input_image_bat.png'), (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, 'canny_step.png'), gray_canny)
    cv2.imwrite(os.path.join(output_dir, 'canny_into_harris.png'), (blurred_heatmap * 255).astype(np.uint8))

    return all_responses.to(device)


def good_features_to_track(input_tensor, max_corners=100, quality_level=0.1, min_distance=5, output_dir='corner_output'):
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


    cv2.imwrite(os.path.join(output_dir, 'input_image_bat.png'), (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, 'good_features_to_track.png'), (blurred_heatmap * 255).astype(np.uint8))
    #cv2.imwrite(os.path.join(output_dir, 'good_features_to_track_step.png'), gray.numpy())

    return all_responses.to(device)

def harris_detector_batch(images, k=0.04, threshold=0.05, output_dir='corner_output'):
        batch_size, channels, height, width = images.shape
        corners_batch = np.zeros((batch_size, height, width), dtype=np.float32)

        device = images.device

        for batch_idx in range(batch_size):
            image = images[batch_idx].to(torch.float32).cpu().numpy()
            combined_corners = np.zeros((height, width), dtype=np.float32)

            for channel_idx in range(channels):
                gray = cv2.normalize(image[channel_idx], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                harris_corners = cv2.cornerHarris(gray, 2, 3, k)
                harris_corners = cv2.dilate(harris_corners, None)
                harris_corners = harris_corners > threshold * harris_corners.max()

                combined_corners = np.maximum(combined_corners, harris_corners)

            corners_batch[batch_idx] = combined_corners
            #cv2.imwrite(os.path.join(output_dir, f'harris_laplace_corners_{batch_idx}.png'), (combined_corners * 255).astype(np.uint8))

        #cv2.imwrite(os.path.join(output_dir, 'input_image_bat.png'), (images[-1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        #cv2.imwrite(os.path.join(output_dir, f'harris_corners_{batch_idx}.png'), (corners_batch[-1] * 255).astype(np.uint8))

        # Convert to tensor and return
        corners_batch = torch.from_numpy(corners_batch)
        # add channel dimension
        corners_batch = corners_batch.unsqueeze(1)

        return corners_batch.to(device)

def add_fast_keypoint_heatmap(tensor_image: torch.Tensor, sigma: float = 2.0):
    """
    Takes a PyTorch tensor image, extracts FAST keypoints using OpenCV,
    creates a heatmap, applies Gaussian blur, and appends it as a new channel.
    
    Args:
        tensor_image (torch.Tensor): Input image tensor (C, H, W) in range [0,1].
        sigma (float): Standard deviation for Gaussian blur.
    
    Returns:
        torch.Tensor: Modified tensor with an extra channel containing the heatmap.
    """
    if tensor_image.dim() != 3:
        raise ValueError("Input tensor must have 3 dimensions (C, H, W)")
    
    # Convert tensor to OpenCV format (H, W, C) and then grayscale
    np_image = (tensor_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY) if np_image.shape[-1] == 3 else np_image
    
    # Use FAST feature detector
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)
    
    # Create heatmap
    h, w = gray.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < w and 0 <= y < h:
            heatmap[y, x] = 1.0
    
    # Apply Gaussian blur
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    
    # Normalize heatmap to [0,1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Convert to PyTorch tensor and add as a new channel
    heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)
    output_tensor = torch.cat([tensor_image, heatmap_tensor], dim=0)
    
    return output_tensor

def get_keypoints_from_predictions(pred_heatmaps, threshold=0.5):
    all_keypoints = []
    for pred_heatmap in pred_heatmaps:
        prob_heatmap = torch.sigmoid(pred_heatmap.clone()).squeeze().numpy()

        local_max = scipy.ndimage.maximum_filter(prob_heatmap, size=5)  # Adjust size
        peaks = (prob_heatmap == local_max) & (prob_heatmap > threshold)

        # Get peak coordinates
        y_coords, x_coords = np.where(peaks)
        keypoints = np.column_stack((x_coords, y_coords))

        all_keypoints.append(keypoints)

    return all_keypoints

def calculate_accuracy(pred_keypoints, target_keypoints, distance_threshold=5, global_image_size=(500, 500)):
    # Compare batchsize number of predicted keypoints to target keypoints
    
    denormalized_target_keypoints = []
    target_kp_amount = 0
    correct_points = 0
    total_distance = 0
    for batch_keypoints in target_keypoints:
        batch_keypoints = [kp.cpu().numpy() * global_image_size[0] for kp in batch_keypoints]
        denormalized_target_keypoints.append(batch_keypoints)
        target_kp_amount += len(batch_keypoints)
    
    num_pred_points = 0
    for i in range(len(pred_keypoints)):
        num_pred_points += len(pred_keypoints[i])
        for idx, pred_point in enumerate(pred_keypoints[i]):
            # Find the closest target point
            closest_target_point = None
            closest_distance = float("inf")
            for target_point in denormalized_target_keypoints[i]:
                distance = np.linalg.norm(pred_point - target_point)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_target_point = target_point

            if closest_distance < distance_threshold:
                correct_points += 1
                denormalized_target_keypoints[i] = [
                        kp for kp in denormalized_target_keypoints[i]
                        if not np.array_equal(kp, closest_target_point)
                    ]
                if len(denormalized_target_keypoints[i]) == 0:
                    break
            
            total_distance += closest_distance

    if num_pred_points == 0:
        print("No keypoints found. Very bad.")
        return -1, -1, 0
        

    return (total_distance / num_pred_points), (correct_points / num_pred_points), (correct_points / target_kp_amount)

def test_detectors(dataset, output_dir, fast_threshold=10):
    dataset_length = len(dataset)
    rand_index = np.random.randint(0, dataset_length)
    sample = dataset[rand_index]
    model_input = sample['image'].unsqueeze(0)

    norm_keypoints = sample['norm_corners']

    image_path = sample['image_path']
    unchanged_image = sample['image'].cpu()
    input_image = sample['image'].permute(1, 2, 0).cpu().numpy()
    cv_image = cv2.imread(image_path)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    
    os.makedirs(output_dir, exist_ok=True)

    # Compute corners using different implementations
    #print("Computing harris corners...")
    harris_corner_map = harris_corner_rgb(cv_image, block_size=1, ksize=1, k=0.03, output_dir=output_dir)
    
    #print("Computing susan corners...")
    #susan_corner_map = susan_corner_detector(cv_image, threshold=15, output_dir=output_dir)
    #print("Computing fast corners...")
    fast_output, fast_keypoints = fast_corner(cv_image, output_dir=output_dir)

    # Calculate accuracy
    #print("Calculating accuracy...")
    fast_distance, fast_accuracy, fast_recall = calculate_accuracy(fast_keypoints, norm_keypoints, global_image_size=input_image.shape[:2], distance_threshold=fast_threshold)
    print(f"FAST: Distance: {fast_distance}, Accuracy: {fast_accuracy}, Recall: {fast_recall}")

    return cv_image, fast_distance, fast_accuracy, fast_recall

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(global_image_size),
    torchvision.transforms.ToTensor()
    #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

print("Loading dataset...")
dataset = LegoKeypointDataset(annotations_folder, img_dir, transform=transforms)
output_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'corner_output')

test_size = 20

dataloader = torch.utils.data.DataLoader(dataset, batch_size=test_size, shuffle=True, collate_fn=collate_fn)

total_distance = 0
total_accuracy = 0
total_recall = 0
val_counter = 0

for i, batch in enumerate(dataloader):
    #print("Computing harris-laplace corners...")
    batch_images = batch['image']
    target_corners = batch['norm_corners']
    # Filter target keypoints by removing (0, 0)
    filtered_target_corners = []
    for corners in target_corners:
        zero_tensor = torch.tensor([0, 0], dtype=torch.float32)
        filtered_corners = [kp for kp in corners if not torch.equal(kp, zero_tensor)]
        filtered_target_corners.append(filtered_corners)
    target_corners = filtered_target_corners
    #harris_laplace_corners = harris_laplace_detector_batch(batch_images, output_dir=output_dir)
    #print("Computing harris-laplace corners with GPU...")
    #harris_laplace_corners_gpu = harris_laplace_detector_gpu(batch_images.to('cuda'), output_dir=output_dir)
    #canny_harris_response = canny_into_harris(batch_images, output_dir=output_dir)
    #susan = susan_corner_detector(batch_images, output_dir=output_dir)
    #good_features = good_features_to_track(batch_images, output_dir=output_dir)
    harris = harris_detector_batch(batch_images, output_dir=output_dir)
    batch_distance, batch_accuracy, batch_recall = calculate_accuracy(get_keypoints_from_predictions(harris.detach().cpu(), threshold=0.5), target_corners, 5, global_image_size)
    print(f"Batch distance: {batch_distance}, Batch accuracy: {batch_accuracy}, Batch recall: {batch_recall}")

    if batch_distance == -1:
        no_points_detected += 1
    else:
        total_distance += batch_distance
        total_accuracy += batch_accuracy
        total_recall += batch_recall

    val_counter += 1

print(f"Average distance: {total_distance / val_counter}, Average accuracy: {total_accuracy / val_counter}, Average recall: {total_recall / val_counter}")


total_recall = 0
for i in range(test_size):
    #cv_image, distance, accuracy, recall = test_detectors(dataset, output_dir, fast_threshold=10)
    #total_recall += recall
    pass

print(f"Average recall: {total_recall / test_size}")

# Save the results separately


print("Saving results...")
#cv2.imwrite(os.path.join(output_dir, 'input_image.png'), cv_image)



