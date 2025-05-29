import numpy as np
import scipy
import torch


def calculate_accuracy(pred_keypoints, target_keypoints, distance_threshold=5, global_image_size=(500, 500)):
    
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
                closest_distance = 0 
            total_distance += closest_distance

    if num_pred_points == 0:
        print("No keypoints found. Very bad.")
        return -1, -1, 0
        

    return (total_distance / num_pred_points), (correct_points / num_pred_points), (correct_points / target_kp_amount)
        


def get_keypoints_from_predictions(pred_heatmaps, threshold=0.5):
    all_keypoints = []
    for pred_heatmap in pred_heatmaps:
        prob_heatmap = torch.sigmoid(pred_heatmap.clone()).squeeze().numpy()

        prob_heatmap = (prob_heatmap - np.min(prob_heatmap)) / (np.max(prob_heatmap) - np.min(prob_heatmap))

        local_max = scipy.ndimage.maximum_filter(prob_heatmap, size=5)
        peaks = (prob_heatmap == local_max) & (prob_heatmap > threshold)

        y_coords, x_coords = np.where(peaks)
        keypoints = np.column_stack((x_coords, y_coords))

        all_keypoints.append(keypoints)

    return all_keypoints

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