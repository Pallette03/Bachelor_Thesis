import os
import json
import shutil


def process_json_file(json_file, images_dir, depth_dir, output_dir):
    with open(json_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                image_filename = record.get("image_path")
                if not image_filename:
                    print("No 'image_path' key in record:", record)
                    continue

                # Combine directory with image file name
                img_path = os.path.join(images_dir, image_filename)
                if os.path.exists(img_path):
                    # Copy the image to the output directory
                    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
                    output_img_path = os.path.join(os.path.join(output_dir, 'images'), os.path.basename(img_path))
                    shutil.copy(img_path, output_img_path)
                else:
                    print(f"Image not found: {img_path}")
                    
                # Check for depth image
                depth_filename = image_filename.replace(".png", "_depth.png")
                depth_path = os.path.join(depth_dir, depth_filename)
                if os.path.exists(depth_path):
                    # Copy the depth image to the output directory
                    os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)
                    output_depth_path = os.path.join(os.path.join(output_dir, 'depth'), os.path.basename(depth_path))
                    shutil.copy(depth_path, output_depth_path)
                else:
                    print(f"Depth image not found: {depth_path}")
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)

if __name__ == "__main__":
    json_file = os.path.join(os.path.dirname(__file__), os.pardir, 'results_with_pred', 'predictions.json')
    images_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'with_clutter', 'validate', 'images', 'rgb')
    depth_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'with_clutter', 'validate', 'images', 'depth')
    output_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'jannik_dir')

    process_json_file(json_file, images_dir, depth_dir, output_dir)