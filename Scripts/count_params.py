import torch
import os

from models.KeyNet.keynet import KeyNet
from models.hourglass.posenet import PoseNet
from models.simpleModel.simple_model import SimpleModel
from unet_model import UNet


model_folder = os.path.join(os.path.dirname(__file__), os.pardir, 'output')

models_to_count = ["95","97","94","96","33","34","35","36","38","41","42","43","58","59","60","61","48","50","51","57"]

unet_params_dict = {}
keynet_params_dict = {}
hourglass_params_dict = {}
simple_model_params_dict = {}

for model_path in os.listdir(model_folder):
    
    if model_path.endswith(".pth") and any(model_path.startswith(prefix) for prefix in models_to_count):
        print(f"Loading model: {model_path}")
        if "UNet" in model_path:
            arch = "UNet"
            model = UNet(n_channels=3, n_classes=1)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(os.path.join(model_folder, model_path), map_location=torch.device('cpu')))
        if "KeyNet" in model_path:
            arch = "KeyNet"
            model = KeyNet(num_filters=8, num_levels=8, kernel_size=5, in_channels=3)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(os.path.join(model_folder, model_path), map_location=torch.device('cpu')))
        if "Hourglass" in model_path:
            arch = "Hourglass"
            model = PoseNet(nstack=4, inp_dim=512, oup_dim=1, bn=False, increase=0, input_image_size=800)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(os.path.join(model_folder, model_path), map_location=torch.device('cpu')))
        if "SimpleModel" in model_path:
            arch = "SimpleModel"
            model = SimpleModel(in_channels=3, out_channels=1)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(os.path.join(model_folder, model_path), map_location=torch.device('cpu')))
    
        print(f"Counting parameters for model: {model_path}")
        
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        if arch == "UNet":
            unet_params_dict[model_path] = pytorch_total_params
        elif arch == "KeyNet":
            keynet_params_dict[model_path] = pytorch_total_params
        elif arch == "Hourglass":
            hourglass_params_dict[model_path] = pytorch_total_params
        elif arch == "SimpleModel":
            simple_model_params_dict[model_path] = pytorch_total_params
    
# Print the number of parameters for each model
for model_path, num_params in unet_params_dict.items():
    print(f"{model_path}: {num_params} parameters")
    
for model_path, num_params in keynet_params_dict.items():
    print(f"{model_path}: {num_params} parameters")

for model_path, num_params in hourglass_params_dict.items():
    print(f"{model_path}: {num_params} parameters")

for model_path, num_params in simple_model_params_dict.items():
    print(f"{model_path}: {num_params} parameters")

