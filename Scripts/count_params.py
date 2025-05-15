import torch
import os

from Scripts.models.KeyNet.keynet import KeyNet
from Scripts.models.hourglass.posenet import PoseNet
from Scripts.models.simpleModel.simple_model import SimpleModel
from Scripts.unet_model import UNet


model_folder = os.path.join(os.path.dirname(__file__), os.pardir, 'output')

models_to_count = ["95","97","94","96","33","34","35","36","38","41","42","43","58","59","60","61","48","50","51","57"]

params_dict = {}

for model_path in os.listdir(model_folder):
    
    if model_path.endswith(".pth") and any(model_path.startswith(prefix) for prefix in models_to_count):
        print(f"Loading model: {model_path}")
        if "UNet" in model_path:
            model = UNet(n_channels=3, n_classes=1)
            model = torch.load(os.path.join(model_folder, model_path), map_location=torch.device('cpu'))
        if "KeyNet" in model_path:
            model = KeyNet(num_filters=8, num_levels=8, kernel_size=5, in_channels=3)
            model = torch.load(os.path.join(model_folder, model_path), map_location=torch.device('cpu'))
        if "Hourglass" in model_path:
            model = PoseNet(nstack=4, inp_dim=512, oup_dim=1, bn=False, increase=0, input_image_size=800)
            model = torch.load(os.path.join(model_folder, model_path), map_location=torch.device('cpu'))
        if "SimpleModel" in model_path:
            model = SimpleModel(in_channels=3, out_channels=1)
            model = torch.load(os.path.join(model_folder, model_path), map_location=torch.device('cpu'))
    
        print(f"Counting parameters for model: {model_path}")
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        params_dict[model_path] = pytorch_total_params
    
# Print the number of parameters for each model
for model_path, num_params in params_dict.items():
    print(f"{model_path}: {num_params} parameters")

