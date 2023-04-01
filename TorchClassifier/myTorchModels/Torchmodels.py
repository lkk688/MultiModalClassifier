#This code should be run in the head node with internet access to download the data
import torch
import torchvision
import torchvision.datasets as datasets
import os
#from torchvision.models import get_model, get_model_weights, list_models
from torchvision.models import get_model, get_model_weights, get_weight, list_models

print(os.getcwd())

#Download model
os.environ['TORCH_HOME'] = '/data/cmpe249-fa22/torchhome/' #setting the environment variable
resnet18 = torchvision.models.resnet18(pretrained=True)
print(resnet18)
resnet50 = torchvision.models.resnet50(pretrained=True)
print(resnet50)
#Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /data/cmpe249-fa22/torchhome/hub/checkpoints/resnet50-19c8e357.pth

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()


print("Torch buildin models:", list_models())
torchvisionallmodels=list_models(module=torchvision.models)
print("Torchvision buildin models:", torchvisionallmodels)

from torchvision.io import read_image
img = read_image("tests/imgdata/rose.jpeg")
model_name="resnet50"
# Step 1: Initialize model with the best available weights
weights_enum = get_model_weights(model_name)
print(weights_enum.IMAGENET1K_V1)
print([weight for weight in weights_enum])
weights = get_weight("ResNet50_Weights.IMAGENET1K_V2")#ResNet50_Weights.DEFAULT
currentmodel=get_model(model_name, weights=weights)#weights="DEFAULT"
currentmodel.eval()
# Step 2: Initialize the inference transforms
preprocess = weights.transforms()
# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)
# Step 4: Use the model and print the predicted category
prediction = currentmodel(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
#The classes of the pre-trained model outputs can be found at weights.meta["categories"]

for model_name in torchvisionallmodels:
    #To get the enum class with all available weights of a specific model you can use either its name:
    weights_enum = get_model_weights(model_name)
    print([weight for weight in weights_enum])
    #
    currentweight=get_weight("ResNet50_Weights.IMAGENET1K_V2")
    print("Current model name:", model_name)
    currentmodel=get_model(model_name, weights="DEFAULT")

#torch hub
model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
weight_enum = torch.hub.load("pytorch/vision", "get_model_weights", name="resnet50")
print([weight for weight in weight_enum])

#save model: https://pytorch.org/tutorials/beginner/saving_loading_models.html
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())