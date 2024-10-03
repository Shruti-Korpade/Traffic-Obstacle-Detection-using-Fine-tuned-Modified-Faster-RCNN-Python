import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Define the same data transformations as used during training
from Code1 import ResNet18

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Specify the path to your test data directory
test_data_dir = "C:/Users/kskor/PycharmProjects/EDI_code/Images/dataset 2/test"

# Create a dataset using the same transformations
test_dataset = ImageFolder(test_data_dir, transform=data_transforms)

# Create a DataLoader for the test dataset
batch_size = 32  # You can adjust this as needed
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the trained model
model = ResNet18()  # Replace with your model if it's different
model.load_state_dict(torch.load('New_cnn.pth'))
model.eval()

# Initialize variables to track accuracy and total samples
correct = 0
total = 0

# Test the model on the test dataset
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print(f"Accuracy on the test dataset: {accuracy:.2f}%")

