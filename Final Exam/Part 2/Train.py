import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
val_dataset = datasets.ImageFolder(root='dataset/test', transform=transform)

# Automatically determine the number of classes from the folder structure
class_labels = train_dataset.classes  # Extract class labels from folder names
num_classes = len(class_labels)

print(f"Class Labels: {class_labels}")
print(f"Number of Classes: {num_classes}")

# Data loaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Display a batch of training images
def imshow(img):
    img = img.numpy().transpose((1, 2, 0))  # Convert Tensor to NumPy array
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Get a batch of training data
images, labels = next(iter(train_loader))
print(images.shape)
# Show images
fig = plt.figure(figsize=(8, 8))
for idx in range(batch_size):
    ax = fig.add_subplot(2, 2, idx + 1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(train_dataset.classes[labels[idx]])

# Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=num_classes).to(device)
# model.load_state_dict(torch.load("model.pth"))
# print(model.fc2)  # Check the output layer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 20
print("\nStarting Training...\n")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {epoch_loss:.4f}")

    # Validation Loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{epochs}], Validation Accuracy: {accuracy:.2f}%\n")

# Save the model
torch.save(model.state_dict(), "model.pth")
print("\nTraining Complete. Model saved as 'model.pth'.")

