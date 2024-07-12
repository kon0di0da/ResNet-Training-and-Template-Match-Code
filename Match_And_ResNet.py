import os
import cv2
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import transforms
import shutil

# Define paths
# TODO: image and model Path
tiff_directory = '../C3 Couplers'  # Directory containing the TIF files
template_path = './coupler.tif'  # Path to the template image
destination_folder = './ClassifiedBad'  # Folder to move classified bad devices
model_path = "./ResNet/resnet_model.pth"  # Path to the ResNet model

# Load the template image in grayscale
template = cv2.imread(template_path, 0)

# Define the ResNet model class
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=False)
        self.fc = nn.Linear(1000, 2)  # 2 classes: good and bad

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

# Load the pre-trained model
model = ResNetModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# Define image preprocessing steps
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL Image
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Iterate through each TIF file and align
for tiff_file in os.listdir(tiff_directory):
    if tiff_file.endswith('.tif') or tiff_file.endswith('.jpg'):
        original_path = os.path.join(tiff_directory, tiff_file)
        destination_path = os.path.join(destination_folder, tiff_file)
        img = cv2.imread(original_path, 0)  # Load the image in grayscale
        image = cv2.imread(original_path, 1)  # Load the image in color

        # Get template image dimensions
        w, h = template.shape[::-1]

        # Perform template matching
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

        # Set a threshold for matching
        threshold = 0.7

        # Find matching locations
        loc = np.where(res >= threshold)

        matches = []
        # Check if there are any matches
        if loc[0].size > 0:
            for pt in zip(*loc[::-1]):
                x, y = pt
                cropped_img = image[y:y + h, x:x + w]
                matches.append(cropped_img)

        if matches:
            # Preprocess the first match
            image_tensor = transform(matches[0]).unsqueeze(0).to(device)

            # Make a prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)

            # Move the file based on the prediction
            if predicted.item() == 0:
                print("Image name:", original_path, "judgement is bad")
                shutil.move(original_path, destination_path)
            else:
                print("Image name:", original_path, "judgement is good")
        else:
            # If no matches, classify as bad
            print("Image name:", original_path, "judgement is bad")
            shutil.move(original_path, destination_path)
