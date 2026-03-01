import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision import datasets

# Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):  # Ensure num_classes matches the trained model
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Activation after first convolution
        x = self.pool(x)           # First pooling
        x = F.relu(self.conv2(x))  # Activation after second convolution
        x = self.pool(x)           # Second pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))    # First fully connected layer
        x = self.fc2(x)            # Output layer
        return x

# Automatically extract class labels from folder structure
data_dir = 'dataset/test'
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

class_labels = sorted(os.listdir(data_dir))  # Get folder names as class labels
print(f"Class Labels: {class_labels}")

# Ensure the number of classes matches the trained model
num_classes = len(class_labels)
if num_classes != 3:
    raise ValueError(f"Mismatch between folder classes ({num_classes}) and trained model (2 classes).")

# Load the model
model = SimpleCNN(num_classes=num_classes)
model_path = 'model.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model.load_state_dict(torch.load(model_path))
model.eval()

#image 1
# Load and transform the image
image_path = 'dataset/test/bridge/1.jpg' # Change this to the path of your image
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor(),         # Convert to Tensor
])
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform prediction
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    max_prob, predicted = torch.max(probabilities, 1)

# Display the image and prediction
image_display = Image.open(image_path)
plt.imshow(image_display)
predicted_class = class_labels[predicted.item()]
plt.title(f'Predicted: {predicted_class}, Confidence: {max_prob.item() * 100:.2f}%')
plt.axis('off')
plt.show()

# Print details to console
print(f"Image Path: {image_path}")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {max_prob.item() * 100:.2f}%")

#image 2
image_path = 'dataset/test/crosswalk/1.jpg' # Change this to the path of your image
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor(),         # Convert to Tensor
])
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform prediction
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    max_prob, predicted = torch.max(probabilities, 1)

# Display the image and prediction
image_display = Image.open(image_path)
plt.imshow(image_display)
predicted_class = class_labels[predicted.item()]
plt.title(f'Predicted: {predicted_class}, Confidence: {max_prob.item() * 100:.2f}%')
plt.axis('off')
plt.show()

# Print details to console
print(f"Image Path: {image_path}")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {max_prob.item() * 100:.2f}%")

#image3
image_path = 'dataset/test/stair/1.jpg' # Change this to the path of your image
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor(),         # Convert to Tensor
])
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform prediction
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    max_prob, predicted = torch.max(probabilities, 1)

# Display the image and prediction
image_display = Image.open(image_path)
plt.imshow(image_display)
predicted_class = class_labels[predicted.item()]
plt.title(f'Predicted: {predicted_class}, Confidence: {max_prob.item() * 100:.2f}%')
plt.axis('off')
plt.show()

# Print details to console
print(f"Image Path: {image_path}")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {max_prob.item() * 100:.2f}%")