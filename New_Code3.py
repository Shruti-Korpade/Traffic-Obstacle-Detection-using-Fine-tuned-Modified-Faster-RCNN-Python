import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations for inference
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)

# Modify the fc layer to match the number of output classes (5)
num_classes = 5
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the trained model weights
model_weights_path = 'resnet50_model.pth'
model.load_state_dict(torch.load(model_weights_path, map_location=device))

# Send the model to the GPU if available
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Load and preprocess the image you want to classify
image_path = 'C:/Users/kskor/PycharmProjects/EDI_code/Images/dataset 2/train/class_1(zebra crossing)/20230113_20_56_56_020_000_zGaC6yksz9cwl3UOetH2K4URrOs2_F_4080_3060_jpg.rf.b16111491a6ef6e9bd6827f74e74956b.jpg'  # Replace with the path to your image
image = Image.open(image_path)
image = data_transform(image).unsqueeze(0)  # Add a batch dimension

# Perform inference
with torch.no_grad():
    image = image.to(device)
    output = model(image)
    _, predicted_class = torch.max(output, 1)

# Map the predicted class index to a label (if you have a label mapping)
label_mapping = {0: 'zebra crossing', 1: 'Traffic sign', 2: 'car', 3: 'bike', 4: 'pedestrian'}
predicted_label = label_mapping[predicted_class.item()]

# Print the result
print(f"Predicted class index: {predicted_class.item()}")
# Uncomment the line below if you have a label mapping
print(f"Predicted class label: {predicted_label}")