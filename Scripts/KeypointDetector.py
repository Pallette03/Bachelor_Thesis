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
from models.hourglass.posenet import PoseNet
from models.hourglass.hourglass import StackedHourglass
from models.simpleModel.simple_model import SimpleModel
from Scripts.models.unet.unet_model import UNet
from models.KeyNet.keynet import KeyNet
import wandb
import argparse


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
        return self.lambda_bce * bce + self.lambda_mse * mse# + self.lambda_focal * focal

class PixelEuclideanLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Compute pixel-wise Euclidean distance
        euclidean_loss = torch.sqrt(torch.sum((torch.sigmoid(pred) - target) ** 2, dim=1))
        return euclidean_loss.mean()

def keypoints_to_heatmap(keypoints, image_size=500, sigma=1.0, gaussian_blur=False, device='cpu'):
    """
    Converts keypoints into a lower-resolution heatmap (e.g., 128×128) for training.
    The heatmap will be upsampled to match the input image size (500×500).
    """
    target_heatmap = torch.zeros((1, image_size, image_size)).to(device)

    if keypoints is None:
        return target_heatmap
    
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

def collate_fn(batch):
    images = [item["image"] for item in batch]
    corners_list = [item["norm_corners"] for item in batch]
    max_corner_amount = max([norm_corners.shape[0] for norm_corners in corners_list])

    # Pad the corners
    for i in range(len(corners_list)):
        corners = corners_list[i]
        pad_amount = max_corner_amount - corners.shape[0]
        pad = np.zeros((pad_amount, 2))
        
        if corners.shape[0] == 0:
            corners_list[i] = pad
        else:
            corners_list[i] = np.concatenate((corners, pad), axis=0)
        
    images = torch.stack(images)
    corners_list = torch.stack([torch.tensor(corners, dtype=torch.float32) for corners in corners_list])

    return {"image": images, "norm_corners": corners_list}

def wait_for_termination(run_handler):
    while True:
        if input("Press 'q' to quit: ") == 'q':
            run_handler.finish()
            break

def calculate_accuracy(pred_keypoints, target_keypoints, distance_threshold=5, global_image_size=(500, 500)):
    # Compare batchsize number of predicted keypoints to target keypoints
    
    denormalized_target_keypoints = []
    target_kp_amount = 0
    correct_points = 0
    total_distance = 0
    for batch_keypoints in target_keypoints:
        if batch_keypoints is None:
            denormalized_target_keypoints.append([])
            continue
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
            if closest_distance == float("inf"):
                closest_distance = 0 # If no target point is found, set distance to 0. needs to be changed
            total_distance += closest_distance

    if num_pred_points == 0:
        print("No keypoints found. Very bad.")
        return -1, -1, 0
        

    return (total_distance / num_pred_points), (correct_points / num_pred_points), (correct_points / target_kp_amount)
        


def get_keypoints_from_predictions(pred_heatmaps, threshold=0.5):
    all_keypoints = []
    for pred_heatmap in pred_heatmaps:
        prob_heatmap = torch.sigmoid(pred_heatmap.clone()).squeeze().numpy()

        # Normalize heatmap to [0, 1]
        prob_heatmap = (prob_heatmap - np.min(prob_heatmap)) / (np.max(prob_heatmap) - np.min(prob_heatmap))

        local_max = scipy.ndimage.maximum_filter(prob_heatmap, size=5)  # Adjust size
        peaks = (prob_heatmap == local_max) & (prob_heatmap > threshold)

        # Get peak coordinates
        y_coords, x_coords = np.where(peaks)
        keypoints = np.column_stack((x_coords, y_coords))

        all_keypoints.append(keypoints)

    return all_keypoints

# Training Loop
def train_model(model, dataloader, epoch_model_path, num_epochs=5, lr=1e-3, global_image_size=(500, 500), gaussian_blur=False, run_handler=None, validataion_params=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    critereon = CombinedLoss()
    #critereon = PixelEuclideanLoss()
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        batch_start_time = time.time()
        model.train()
        total_corner_loss = 0.0
        counter = 0
        epoch_start_time = time.time()

        for batch in dataloader:

            images = batch["image"].to(device)  # Shape: [batch_size, 3, H, W]
            target_corners = batch["norm_corners"].to(device)  # Shape: [batch_size, num_corners_in_batch, 2]

            filtered_target_corners = []
            for corners in target_corners:
                zero_tensor = torch.tensor([0, 0], dtype=torch.float32).to(device)
                filtered_corners = [kp for kp in corners if not torch.equal(kp, zero_tensor)]
                
                if len(filtered_corners) == 0:
                    filtered_target_corners.append(None)
                else:
                    filtered_target_corners.append(filtered_corners)
                    
            target_corners = filtered_target_corners

            predicted_corners = model(images)

            batch_len = len(images)
            
            corner_loss = heatmap_loss(predicted_corners, target_corners, image_size=global_image_size[0], device=device, critereon=critereon, gaussian_blur=gaussian_blur)
            last_loss = corner_loss.item()


            # Backward pass
            optimizer.zero_grad()
            corner_loss.backward()
            optimizer.step()
            #scheduler.step()
            
            # Accumulate losses for logging
            total_corner_loss += corner_loss.item()
          
            counter += 1
            # Check the progress through the batch and print every 5 percent
            time_since_start = time.time() - epoch_start_time
            remaining_time = (time_since_start / (counter / len(dataloader))) - time_since_start
            
            if counter % (len(dataloader) // 20) == 0:

                # pred_heatmap = predicted_corners[-1].squeeze(0).detach().cpu().numpy()


                # # Convert to probability using sigmoid
                # prob_heatmap = torch.sigmoid(torch.tensor(pred_heatmap)).detach().cpu().numpy()

                # # Save a heatmap where every pixel gets a color from black to red according to its value
                # plt.imshow(prob_heatmap, cmap='Reds', interpolation='nearest')
                # plt.title("Predicted Heatmap (Black to Red)")
                # plt.colorbar()
                # plt.savefig("predicted_heatmap_black_to_red.png")
                # plt.close()

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
            f1_score = validate_model(model, **validataion_params)
            torch.cuda.empty_cache()
        else:
            f1_score = None

    return model, f1_score

def validate_model(model, dataloader, global_image_size, gaussian_blur=False, threshold=0.5, distance_threshold=10, run_handler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validating on {device}")
    torch.cuda.empty_cache()

    with torch.no_grad():
        model = model.to(device)
        total_corner_loss = 0.0
        critereon = CombinedLoss()
        val_counter = 0
        val_batch_start_time = time.time()
        average_distance = 0
        accuracy = 0
        recall = 0
        no_points_detected = 0

        for batch in dataloader:
            
            images = batch["image"].to(device)  # Shape: [batch_size, 3, H, W]
            target_corners = batch["norm_corners"].to(device)  # Shape: [batch_size, 1, H, W]

            filtered_target_corners = []
            for corners in target_corners:
                zero_tensor = torch.tensor([0, 0], dtype=torch.float32).to(device)
                filtered_corners = [kp for kp in corners if not torch.equal(kp, zero_tensor)]
                
                if len(filtered_corners) == 0:
                    filtered_target_corners.append(None)
                else:
                    filtered_target_corners.append(filtered_corners)

            target_corners = filtered_target_corners

            predicted_corners = model(images)
            
            # Use normal MSE loss for now but ignore 0
            corner_loss = corner_loss = heatmap_loss(predicted_corners, target_corners, image_size=global_image_size[0], device=device, critereon=critereon, gaussian_blur=gaussian_blur)
            val_last_loss = corner_loss.item()
            
            # Accumulate losses for logging
            total_corner_loss += corner_loss.item()

            batch_average_distance, batch_accuracy, batch_recall = calculate_accuracy(get_keypoints_from_predictions(predicted_corners.detach().cpu(), threshold=threshold), target_corners, distance_threshold, global_image_size)
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
        
        if no_points_detected == len(dataloader):
            print("No keypoints detected in any of the images. Very bad.")
            average_distance = -1
            accuracy = -1
            f1_score = -1
        else: 
            average_distance /= (len(dataloader) - no_points_detected)
            accuracy /= (len(dataloader) - no_points_detected)
            
        recall /= (len(dataloader))
        f1_score = 2 * (accuracy * recall) / (accuracy + recall)
        run_handler.log({"Validation Loss": total_corner_loss / len(dataloader), "Validation Average Distance": average_distance, "Validation Accuracy": accuracy, "Recall": recall, "F1 Score": f1_score})
        print(f"Validation Loss: {total_corner_loss / len(dataloader):.4f}, Validation Average Distance: {average_distance:.4f}, Validation Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
    
    torch.cuda.empty_cache()
    
    
    return f1_score


def main(params):
    
    run = wandb.init(project="dynamic_corner_detector", config=params, reinit=True)
    
    with_validate = True

    #Clear vram memory
    torch.cuda.empty_cache()
    
    architecture = run.config["model"]
    dataset = run.config["dataset"]
    batch_size = run.config["batch_size"]
    val_batch_size = run.config["val_batch_size"]
    global_image_size = run.config["global_image_size"]
    learning_rate = run.config["learning_rate"]
    num_epochs = run.config["num_epochs"]
    gaussian_blur = run.config["gaussian_blur"]
    threshold = run.config["post_processing_threshold"]
    distance_threshold = run.config["distance_threshold"]
    num_channels = run.config["num_channels"]
    num_levels = run.config["feature_extractor_lvl_amount"]
    start_from_checkpoint = run.config["start_from_checkpoint"]
    num_stacks = run.config["hourglass_stacks"]
    only_validate = run.config["only_validate"]
    val_model_path = run.config["val_model_path"]

    # Paths
    model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', (architecture + '_' + dataset + '.pth'))
    epoch_model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', (architecture + '_' + dataset + '_epoch.pth'))
    train_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', dataset, 'train')
    validate_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', dataset, 'validate')

    

    print(f"Paths: {model_path}, {epoch_model_path}, {train_dir}, {validate_dir}")

    

    if num_channels == 1:
        transform_1 = transforms.Compose([
            transforms.Resize(global_image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    else:
        transform_1 = transforms.Compose([
            transforms.Resize(global_image_size),
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    if architecture == "UNet":
        model = UNet(n_channels=num_channels, n_classes=1)
    elif architecture == "KeyNet":
        model = KeyNet(num_filters=8, num_levels=num_levels, kernel_size=5, in_channels=num_channels)
    elif architecture == "SimpleModel":
        model = SimpleModel(in_channels=num_channels, out_channels=1)
    elif architecture == "Hourglass":
        model = StackedHourglass(num_stacks=2, num_channels=256)
    elif architecture == "Hourglass_Github":
        model = PoseNet(nstack=num_stacks, inp_dim=512, oup_dim=1, bn=False, increase=0, input_image_size=global_image_size[0])


    if start_from_checkpoint:
        print("Loading model from checkpoint...")
        model.load_state_dict(torch.load(epoch_model_path))

    
    print(f"{torch.cuda.device_count()} GPUs available")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)

    model = nn.DataParallel(model)
    # Train the model
    
    
    

    if only_validate:
        print("Validating the model...")
        val_dataset = LegoKeypointDataset(os.path.join(validate_dir, 'annotations'), os.path.join(validate_dir, 'images', 'rgb'), transform=transform_1)
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn)
        val_model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', val_model_path)
        
        validataion_params = {
            "dataloader": val_dataloader,
            "global_image_size": global_image_size,
            "gaussian_blur": gaussian_blur,
            "threshold": threshold,
            "distance_threshold": distance_threshold,
            "run_handler": run
        }
        
        print("Validating the model...")
        model.load_state_dict(torch.load(val_model_path))
        f1_score = validate_model(model, **validataion_params)
        print(f"F1 Score: {f1_score:.4f}")
        run.finish()
        return
        
    # Dataset and DataLoader
    print("Loading training dataset...")
    train_dataset = LegoKeypointDataset(os.path.join(train_dir, 'annotations'), os.path.join(train_dir, 'images', 'rgb'), transform=transform_1)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    if with_validate:
        print("Loading validation dataset...")
        val_dataset = LegoKeypointDataset(os.path.join(validate_dir, 'annotations'), os.path.join(validate_dir, 'images', 'rgb'), transform=transform_1)
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn)
        
        validataion_params = {
            "dataloader": val_dataloader,
            "global_image_size": global_image_size,
            "gaussian_blur": gaussian_blur,
            "threshold": threshold,
            "distance_threshold": distance_threshold,
            "run_handler": run
        }
        
        # print("Validating the model...")
        # model.load_state_dict(torch.load(epoch_model_path))
        # validate_model(model, **validataion_params)
        print("Training the model...")
        model, f1_score = train_model(model, train_dataloader, epoch_model_path, num_epochs=num_epochs, lr=learning_rate, global_image_size=global_image_size, gaussian_blur=gaussian_blur, run_handler=run, validataion_params=validataion_params)
    
    else:
        print("Training the model...")
        model, f1_score = train_model(model, train_dataloader, epoch_model_path, num_epochs=num_epochs, lr=learning_rate, global_image_size=global_image_size, gaussian_blur=gaussian_blur, run_handler=run, validataion_params=None)

    # Save the model
    print("Saving the model...")
    torch.save(model.state_dict(), model_path)
    
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)
    run.log_artifact(artifact)
    
    run.finish()
    print("Model saved.")
    
    return model, f1_score



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a keypoint detection model")
    parser.add_argument("--model", type=str, default="UNet", help="Model architecture to use (UNet, KeyNet, SimpleModel, Hourglass, Hourglass_Github)")
    parser.add_argument("--dataset", type=str, default="gaussian_noise", help="Dataset to use (with_clutter, without_clutter)")
    parser.add_argument("--batch_size", type=int, default=17, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=17, help="Batch size for validation")
    parser.add_argument("--learning_rate", type=float, default=0.00342, help="Learning rate for the optimizer")
    parser.add_argument("--global_image_size", type=int, default=800, help="Global image size for training")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs to train the model")
    parser.add_argument("--num_channels", type=int, default=3, help="Number of channels in the input images")
    parser.add_argument("--gaussian_blur", type=bool, default=True, help="Whether to apply Gaussian blur to the heatmaps")
    parser.add_argument("--start_from_checkpoint", type=bool, default=False, help="Whether to start training from a checkpoint")
    parser.add_argument("--post_processing_threshold", type=float, default=0.4, help="Threshold for post-processing the heatmaps")
    parser.add_argument("--distance_threshold", type=float, default=5, help="Distance threshold for keypoint matching")
    parser.add_argument("--feature_extractor_lvl_amount", type=int, default=3, help="Number of levels in the feature extractor")
    parser.add_argument("--hourglass_stacks", type=int, default=4, help="Number of stacks in the hourglass model")
    parser.add_argument("--only_validate", type=bool, default=False, help="Whether to only validate the model")
    parser.add_argument("--val_model_path", type=str, default="dynamic_corner_detector.pth", help="model used if only validation is set to True")
    
    params = parser.parse_args()
    params = vars(params)  # Convert Namespace to dict
    params['global_image_size'] = (params['global_image_size'], params['global_image_size'])
    main(params)