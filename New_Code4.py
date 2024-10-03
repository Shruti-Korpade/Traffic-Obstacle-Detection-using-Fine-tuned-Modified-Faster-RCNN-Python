import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Define the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Load and preprocess the image you want to detect objects in
image_path = 'C:/Users/kskor/PycharmProjects/EDI_code/Images/dataset 2/train/class_5(pedestrain)/image (10).jpg'
image = Image.open(image_path).convert("RGB")
image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():
    prediction = model(image_tensor)

# Define the indices of your classes of interest
classes_of_interest = [0, 1, 2, 3, 4]  # Replace with your class indices

# Get bounding boxes, labels, and scores
boxes = prediction[0]['boxes']
labels = prediction[0]['labels']
scores = prediction[0]['scores']

# Set a score threshold for detected objects
score_threshold = 0.7

# Create a PIL image to draw bounding boxes
draw = ImageDraw.Draw(image)

# Iterate through the detected objects
for box, label, score in zip(boxes, labels, scores):
    if score > score_threshold and label.item() in classes_of_interest:
        box = [round(i, 2) for i in box.tolist()]  # Round box coordinates
        draw.rectangle(box, outline="red", width=3)  # Draw bounding box
        label_text = f"Class {label.item()}"
        draw.text((box[0], box[1]), label_text, fill="red")

# Save or display the image with bounding boxes
image.show()




#
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import cv2
# import numpy as np
#
# # Define the device (GPU or CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Define data transformations for inference
# data_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
# # Load the pre-trained ResNet-50 model
# model = models.resnet50(pretrained=True)
#
# # Modify the fc layer to match the number of output classes (5)
# num_classes = 5
# model.fc = nn.Linear(model.fc.in_features, num_classes)
#
# # Load the trained model weights
# model_weights_path = 'resnet50_model.pth'
# model.load_state_dict(torch.load(model_weights_path, map_location=device))
#
# # Send the model to the GPU if available
# model = model.to(device)
# model.eval()  # Set the model to evaluation mode
#
# # Load and preprocess the image you want to classify
# image_path = 'C:/Users/kskor/PycharmProjects/EDI_code/Images/dataset 2/train/class_4(bike)/Bike-1-_jpg.rf.d93c021fb174375df5f691f32b8075f4.jpg'  # Replace with the path to your image
# image = Image.open(image_path)
# original_image = image.copy()  # Make a copy to draw the bounding box on
#
# # Apply the data transformation
# image = data_transform(image).unsqueeze(0)  # Add a batch dimension
#
# # Perform inference
# with torch.no_grad():
#     image = image.to(device)
#     output = model(image)
#     _, predicted_class = torch.max(output, 1)
#
# # Map the predicted class index to a label (if you have a label mapping)
# label_mapping = {0: 'zebra crossing', 1: 'Traffic sign', 2: 'car', 3: 'bike', 4: 'pedestrian'}
# predicted_label = label_mapping[predicted_class.item()]
#
# # Print the result
# print(f"Predicted class index: {predicted_class.item()}")
# print(f"Predicted class label: {predicted_label}")
#
# # Draw a bounding box on the original image
# # Replace these coordinates with the coordinates of the bounding box you want to draw
# # For simplicity, I'm drawing a bounding box around the entire image
# bbox = [(0, 0), (original_image.width, original_image.height)]
# color = (0, 255, 0)  # Green color
# thickness = 2
# image_with_bbox = np.array(original_image)
# cv2.rectangle(image_with_bbox, bbox[0], bbox[1], color, thickness)
#
# # Show the image with the bounding box
# cv2.imshow('Image with Bounding Box', image_with_bbox)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
