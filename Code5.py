import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Load the pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.load_state_dict(torch.load('resnet50_model.pth'))
model.eval()

# Define the transformations to be applied to your image
transform = T.Compose([T.ToTensor()])

# Load and preprocess your image
image = Image.open('C:/Users/kskor/PycharmProjects/EDI_code/Images/dataset 2/train/class_3(car)/Car-1-_jpg.rf.920bc499743589c5aba6a9ac74080a66.jpg')
image_tensor = transform(image)

# Make predictions
with torch.no_grad():
    predictions = model([image_tensor])

# Define a threshold for object detection confidence
threshold = 0.5

# Get the bounding box coordinates, labels, and scores
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

# Create a PIL ImageDraw object to draw bounding boxes on the image
draw = ImageDraw.Draw(image)

# Iterate through the detections and draw bounding boxes for each object
for i in range(len(boxes)):
    if scores[i] > threshold:
        box = boxes[i].cpu().numpy()
        label = labels[i].cpu().numpy()
        score = scores[i].cpu().numpy()
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='green', width=3)
        draw.text((box[0], box[1]), f'Label: {label}, Score: {score}', fill='red')

# Display the image with bounding boxes
plt.imshow(image)
plt.axis('off')
plt.show()