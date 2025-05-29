# Bachelor Thesis Project

This repository contains the code and resources for my Bachelor's thesis project, which involves dataset generation, model training, and testing.

## Dataset Generation

In order to generate a new dataset, the parameters in blender_wrapper.py need to be adjusted to the desired settings. Blender needs to be installed, and the blender.exe path needs to be input at the top after "BLENDER_PATH". After installing the requirements from requirements.txt the script can generate a dataset.

```bash
pip install -r requirements.txt
```

Currently, there are 100 Clutter objects in the google_scanned_objects folder. In case more are needed, they can be downloaded from https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research

## Model Training

To train a model execute the KeypointDetector.py script. The appropriate CUDA drivers need to be installed separately.

The script accepts the following command-line arguments:

| Argument                     | Type    | Default                     | Description                                                                 |
|------------------------------|---------|-----------------------------|-----------------------------------------------------------------------------|
| `--model`                    | `str`   | `"UNet"`                    | Model architecture to use (`UNet`, `KeyNet`, `SimpleModel`, `Hourglass_Github`) |
| `--dataset`                  | `str`   | `"gaussian_noise"`          | Dataset to use                      |
| `--batch_size`               | `int`   | `17`                        | Batch size for training                                                     |
| `--val_batch_size`           | `int`   | `17`                        | Batch size for validation                                                   |
| `--learning_rate`            | `float` | `0.00342`                   | Learning rate for the optimizer                                             |
| `--global_image_size`        | `int`   | `800`                       | Global image size for training                                              |
| `--num_epochs`               | `int`   | `15`                        | Number of epochs to train the model                                         |
| `--num_channels`             | `int`   | `3`                         | Number of channels in the input images                                      |
| `--gaussian_blur`            | `bool`  | `True`                      | Whether to apply Gaussian blur to the heatmaps                              |
| `--start_from_checkpoint`    | `bool`  | `False`                     | Whether to start training from a checkpoint                                 |
| `--post_processing_threshold`| `float` | `0.4`                       | Threshold for post-processing the heatmaps                                  |
| `--distance_threshold`       | `float` | `5`                         | Distance threshold for keypoint matching                                    |
| `--feature_extractor_lvl_amount` | `int` | `3`                     | Number of levels in the Key.Net feature extractor                                   |
| `--hourglass_stacks`         | `int`   | `4`                         | Number of stacks in the hourglass model                                     |
| `--only_validate`            | `bool`  | `False`                     | Whether to only validate the model                                          |
| `--val_model_path`           | `str`   | `"dynamic_corner_detector.pth"` | Model path used if only validation is set to True                       |



