# run_with_params.py
import subprocess
import os

BLENDER_PATH = "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"
BLEND_FILE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "hdri_setup.blend")
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "dataset_generation.py")

# Example list of parameter sets
param_sets = [
    {
        "--rendered_images_amount": 200,
        "--output_folder": "//datasets/clutter",
        "--hdri_folder": "//HDRI",
        "--gaussian_noise": False,
        "--salt_and_pepper_noise": False,
        "--depth_output": False,
        "--add_clutter": False,
        "--fill_to_max_items": False,
        "--will_render_image": True,
        "--filter_occlusions": False,
        "--min_items": 20,
        "--max_items": 30,
        "--min_clutter_items": 5,
        "--max_clutter_items": 15,
        "--min_z": 0.3,
        "--max_z": 1.5,
        "--rendered_image_resolution": 1000
    }
]

def build_command(params):
    cmd = [
        BLENDER_PATH,
        BLEND_FILE_PATH,
        "--background",
        "--python", SCRIPT_PATH,
        "--"
    ]

    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(key)
        else:
            cmd.extend([key, str(value)])
    return cmd

for i, param_set in enumerate(param_sets):
    print(f"\n--- Running configuration {i+1} ---")
    command = build_command(param_set)
    subprocess.run(command, shell=True)
