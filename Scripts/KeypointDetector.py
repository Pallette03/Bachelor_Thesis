import datetime
import os
import sys
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from LegoKeypointDataset import LegoKeypointDataset
import torchvision.transforms as transforms
import time
from unet_model import UNet
import wandb
import threading


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)  # Convert logits to probabilities
        target = target.float()
        
        # Compute focal weight
        focal_weight = self.alpha * (1 - pred_prob) ** self.gamma * target + (1 - self.alpha) * pred_prob ** self.gamma * (1 - target)
        
        # Compute BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
        
        # Apply focal weighting
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()
    
class CombinedLoss(nn.Module):
    def __init__(self, lambda_bce=1.0, lambda_mse=0.1, lambda_focal=1.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.lambda_bce = lambda_bce
        self.lambda_mse = lambda_mse
        self.lambda_focal = lambda_focal

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        mse = self.mse_loss(torch.sigmoid(pred), target)
        focal = self.focal_loss(pred, target)
        return self.lambda_bce * bce + self.lambda_mse * mse + self.lambda_focal * focal

def keypoints_to_heatmap(keypoints, image_size=500, sigma=1.0, gaussian_blur=False, device='cpu'):
    """
    Converts keypoints into a lower-resolution heatmap (e.g., 128×128) for training.
    The heatmap will be upsampled to match the input image size (500×500).
    """
    target_heatmap = torch.zeros((1, image_size, image_size)).to(device)

    for (x, y) in keypoints:
        x = x * image_size
        y = y * image_size
        x, y = int(x), int(y)  # Scale keypoints
        
        if gaussian_blur:
            if 0 <= x < image_size and 0 <= y < image_size:
                for i in range(-1, 2):  # Small 5×5 Gaussian
                    for j in range(-1, 2):
                        xi, yj = x + i, y + j
                        if 0 <= xi < image_size and 0 <= yj < image_size:
                            exponent = torch.tensor(-((i**2 + j**2) / (2 * sigma**2)), dtype=torch.float32).to(device)
                            target_heatmap[0, yj, xi] += torch.exp(exponent).to(device)
        else:
            target_heatmap[0, y, x] = 1

    
    # if not os.path.exists(os.path.join(os.path.dirname(__file__), os.pardir, 'heatmap.png')):
    #     heatmap = target_heatmap.squeeze().detach().cpu().numpy()
    #     plt.imshow(heatmap, cmap='Reds', interpolation='nearest')
    #     plt.title("Target Heatmap (Black to Red)")
    #     plt.colorbar()
    #     plt.savefig("heatmap.png")
    #     plt.close()

    return target_heatmap


def heatmap_loss(pred_heatmaps, keypoints_list, image_size=500, device='cpu', critereon=None, gaussian_blur=False):

    target_heatmaps = torch.stack([keypoints_to_heatmap(kp, image_size=image_size, device=device, gaussian_blur=gaussian_blur) for kp in keypoints_list]).to(device)

    loss = critereon(pred_heatmaps, target_heatmaps)

    #loss = torch.abs(loss)

    return loss

def extract_coordinates(heatmap, threshold=0.5):
    heatmap = heatmap.squeeze().detach().cpu().numpy()  # Remove batch & channel dims
    corners = torch.nonzero(heatmap > threshold)  # Get (y, x) positions
    corners = corners.numpy() * 16  # Upscale to original image size (500x500)
    return corners  # List of (y, x) positions

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

def wait_for_termination(run_handler):
    while True:
        if input("Press 'q' to quit: ") == 'q':
            run_handler.finish()
            break

def calculate_accuracy(pred_keypoints, target_keypoints, distance_threshold=10, global_image_size=(500, 500)):
    # Compare batchsize number of predicted keypoints to target keypoints
    if len(pred_keypoints) != len(target_keypoints):
        raise ValueError("Batch sizes of predicted and target keypoints do not match")
    
    denormalized_target_keypoints = []
    target_kp_amount = 0
    for batch_keypoints in target_keypoints:
        batch_keypoints = [kp * global_image_size[0] for kp in batch_keypoints]
        denormalized_target_keypoints.append(batch_keypoints)
        target_kp_amount += len(batch_keypoints)
    
    pred_target_distances = []
    for i in range(len(pred_keypoints)):
        for pred_point in pred_keypoints[i]:
            # Find the closest target point
            closest_target_point = None
            closest_distance = float("inf")
            for target_point in denormalized_target_keypoints[i]:
                distance = np.linalg.norm(pred_point - target_point)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_target_point = target_point

            pred_target_distances.append(closest_distance)

    if len(pred_target_distances) == 0:
        print("No keypoints found. Very bad.")
        return -1, -1, 0

    # Calculate the average distance between the predicted and target points
    total_distance = 0
    correct_points = 0
    for distance in pred_target_distances:
        total_distance += distance
        if distance < distance_threshold:
            correct_points += 1 

    return (total_distance / len(pred_target_distances)), (correct_points / len(pred_target_distances)), (correct_points / target_kp_amount)
        


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

# Training Loop
def train_model(model, dataloader, epoch_model_path, num_epochs=5, lr=1e-3, global_image_size=(500, 500), gaussian_blur=False, run_handler=None, termination_thread=None, validataion_params=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    critereon = CombinedLoss(lambda_bce=1.0, lambda_mse=0.1, lambda_focal=1.0, alpha=0.25, gamma=2.0)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        batch_start_time = time.time()
        model.train()
        total_corner_loss = 0.0
        counter = 0
        epoch_start_time = time.time()

        for batch in dataloader:
            if termination_thread.is_alive() == False:
                sys.exit(0)

            images = batch["image"].to(device)  # Shape: [batch_size, 3, H, W]
            target_corners = batch["norm_corners"].to(device)  # Shape: [batch_size, num_corners_in_batch, 2]

            predicted_corners = model(images)

            batch_len = len(images)
            
            corner_loss = heatmap_loss(predicted_corners, target_corners, image_size=global_image_size[0], device=device, critereon=critereon, gaussian_blur=gaussian_blur)
            last_loss = corner_loss.item()


            # Backward pass
            optimizer.zero_grad()
            corner_loss.backward()
            optimizer.step()
            
            # Accumulate losses for logging
            total_corner_loss += corner_loss.item()
          
            counter += 1
            # Check the progress through the batch and print every 5 percent
            time_since_start = time.time() - epoch_start_time
            remaining_time = (time_since_start / (counter / len(dataloader))) - time_since_start
            
            if counter % (len(dataloader) // 20) == 0:
                log_string = f"[{datetime.datetime.now()}] At Batch {counter}/{len(dataloader)} for Epoch {epoch + 1} taking {time.time() - batch_start_time:.2f} seconds since last checkpoint. Last Loss: {last_loss:.4f}. Progress: {counter / len(dataloader) * 100:.2f}%. Approx. Time left: {remaining_time:.2f} seconds"
                run_handler.log({"Batch Progress": f"{counter}/{len(dataloader)}", "Time since last checkpoint": (time.time() - batch_start_time), "Last Loss": last_loss, "Progress": (counter / len(dataloader) * 100), "Approx. Time left": remaining_time})
                print(log_string)
                batch_start_time = time.time()


        #Clear vram memory before validation
        torch.cuda.empty_cache()

        # Log epoch stats
        print(f"Took {time.time() - start_time:.2f} seconds for epoch {epoch + 1}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Heatmap Loss: {total_corner_loss / len(dataloader):.4f}")
        
        # Save model after each epoch
        print(f"Saving model to {epoch_model_path}")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Saved model to {epoch_model_path}")

        if validataion_params:
            validate_model(model, **validataion_params)

    return model

def validate_model(model, dataloader, global_image_size, gaussian_blur=False, threshold=0.5, distance_threshold=10, termination_thread=None, run_handler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validating on {device}")
    torch.cuda.empty_cache()

    with torch.no_grad():
        model = model.to(device)
        total_corner_loss = 0.0
        critereon = CombinedLoss(lambda_bce=1.0, lambda_mse=0.1, lambda_focal=1.0, alpha=0.25, gamma=2.0)
        val_counter = 0
        val_batch_start_time = time.time()
        average_distance = 0
        accuracy = 0
        recall = 0
        no_points_detected = 0

        for batch in dataloader:
            if termination_thread.is_alive() == False:
                sys.exit(0)
            
            images = batch["image"].to(device)  # Shape: [batch_size, 3, H, W]
            target_corners = batch["norm_corners"].to(device)  # Shape: [batch_size, 1, H, W]
            #target_offsets = batch["offsets"].to(device)  # Shape: [batch_size, 2, H, W]
            #mask = batch["mask"].to(device)  # Shape: [batch_size, 1, H, W], 1 for valid corner locations
            predicted_corners = model(images)#, predicted_offsets
            
            # Use normal MSE loss for now but ignore 0
            corner_loss = corner_loss = heatmap_loss(predicted_corners, target_corners, image_size=global_image_size[0], device=device, critereon=critereon, gaussian_blur=gaussian_blur)
            val_last_loss = corner_loss.item()
            
            # Accumulate losses for logging
            total_corner_loss += corner_loss.item()

            batch_average_distance, batch_accuracy, batch_recall = calculate_accuracy(get_keypoints_from_predictions(predicted_corners.detach().cpu(), threshold=threshold), target_corners.detach().cpu().numpy(), distance_threshold, global_image_size)
            if batch_average_distance == -1:
                no_points_detected += 1
            else:
                average_distance += batch_average_distance
                accuracy += batch_accuracy
                recall += batch_recall

            val_counter += 1
            # Check the progress through the batch and print every 5 percent
            if (val_counter % (len(dataloader) // 20) == 0) or (val_counter == len(dataloader)):
                print(f"[{datetime.datetime.now()}] At Batch {val_counter}/{len(dataloader)} for Validation taking {time.time() - val_batch_start_time:.2f} seconds since last checkpoint. Last Loss: {val_last_loss:.4f}")
                val_batch_start_time = time.time()
        
        average_distance /= (len(dataloader) - no_points_detected)
        accuracy /= (len(dataloader) - no_points_detected)
        recall /= (len(dataloader))
        run_handler.log({"Validation Loss": total_corner_loss / len(dataloader), "Validation Average Distance": average_distance, "Validation Accuracy": accuracy, "Recall": recall})
        print(f"Validation Loss: {total_corner_loss / len(dataloader):.4f}")
    
    torch.cuda.empty_cache()
    return total_corner_loss / len(dataloader)

if __name__ == "__main__":

    run = wandb.init(
        project='lego-keypoint-detection', 
        entity='pallette-personal', 
        job_type='train',
        config={
            "model": "UNet",
            "dataset": "cropped_objects",
            "batch_size": 6,
            "val_batch_size": 6,
            "learning_rate": 1e-4,
            "global_image_size": (650, 650),
            "num_epochs": 10,
            "gaussian_blur": True,
            "post_processing_threshold": 0.5,
            "distance_threshold": 10
            }
        )

    #Clear vram memory
    torch.cuda.empty_cache()

    termination_thread = threading.Thread(target=wait_for_termination, args=(run,), daemon=True)
    termination_thread.start()

    # Paths
    model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', 'dynamic_corner_detector.pth')
    epoch_model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', 'dynamic_corner_detector_epoch.pth')
    train_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'train')
    validate_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'validate')

    with_validate = True

    print(f"Paths: {model_path}, {epoch_model_path}, {train_dir}, {validate_dir}")

    batch_size = run.config["batch_size"]
    val_batch_size = run.config["val_batch_size"]
    global_image_size = run.config["global_image_size"]
    learning_rate = run.config["learning_rate"]
    num_epochs = run.config["num_epochs"]
    gaussian_blur = run.config["gaussian_blur"]
    threshold = run.config["post_processing_threshold"]
    distance_threshold = run.config["distance_threshold"]

    transform_1 = transforms.Compose([
        transforms.Resize(global_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    

    # Model, Optimizer, and Loss
    model = UNet(n_channels=3, n_classes=1)

    # Train the model
    
    
    # Dataset and DataLoader
    print("Loading training dataset...")
    train_dataset = LegoKeypointDataset(os.path.join(train_dir, 'annotations'), os.path.join(train_dir, 'images'), transform=transform_1)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    if with_validate:
        print("Loading validation dataset...")
        val_dataset = LegoKeypointDataset(os.path.join(validate_dir, 'annotations'), os.path.join(validate_dir, 'images'), transform=transform_1)
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn)
        
        validataion_params = {
            "dataloader": val_dataloader,
            "global_image_size": global_image_size,
            "gaussian_blur": gaussian_blur,
            "threshold": threshold,
            "distance_threshold": distance_threshold,
            "termination_thread": termination_thread,
            "run_handler": run
        }
        
        print("Validating the model...")
        model.load_state_dict(torch.load(epoch_model_path))
        validate_model(model, **validataion_params)
        # print("Training the model...")
        # model = train_model(model, train_dataloader, epoch_model_path, num_epochs=num_epochs, lr=learning_rate, global_image_size=global_image_size, gaussian_blur=gaussian_blur, run_handler=run, termination_thread=termination_thread, validataion_params=validataion_params)
    
    else:
        print("Training the model...")
        model = train_model(model, train_dataloader, epoch_model_path, num_epochs=num_epochs, lr=learning_rate, global_image_size=global_image_size, gaussian_blur=gaussian_blur, run_handler=run, termination_thread=termination_thread, validataion_params=None)

    # Save the model
    print("Saving the model...")
    torch.save(model.state_dict(), model_path)
    run.finish()

        