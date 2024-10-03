import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
import numpy as np
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Load your own ResNet-18 model for classification
# Replace 'YourCustomResNet' with the actual name of your ResNet model class
from Code1 import ResNet18

# Load the trained weights of your ResNet model
model_path = 'New_cnn.pth'
classification_model = ResNet18()
classification_model.load_state_dict(torch.load(model_path))
classification_model.eval()

# Load a pre-trained object detection model (e.g., Faster R-CNN)
backbone = torchvision.models.resnet18(pretrained=True)
backbone.out_channels = 2048
rpn_anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)
object_detection_model = FasterRCNN(
    backbone,
    num_classes=91,  # You may need to adjust this based on your dataset
    rpn_anchor_generator=rpn_anchor_generator
)
object_detection_model.eval()

# Input Image Processing
def preprocess_image(image):
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)

image_path = 'C:/Users/kskor/PycharmProjects/EDI_code/Images/dataset 2/tesT/class_1(zebra crossing)/20230114_15_06_58_141_000_JEVDSuoPy1SxjnISG6v8523EcMF3_T_4080_1836_jpg.rf.8b5b89d6508d6f0159c4a1513d97cb97.jpg'
input_image = Image.open(image_path)
input_tensor = preprocess_image(input_image)

# Object Detection
with torch.no_grad():
    prediction = object_detection_model(input_tensor)

# Extract bounding box coordinates and class labels
boxes = prediction[0]['boxes'].cpu().numpy()
labels = prediction[0]['labels'].cpu().numpy()
scores = prediction[0]['scores'].cpu().numpy()

# Classification
for box, label, score in zip(boxes, labels, scores):
    x1, y1, x2, y2 = box.astype(int)
    roi = input_image.crop((x1, y1, x2, y2))
    roi_tensor = preprocess_image(roi)

    # Classify the ROI using your custom ResNet model
    with torch.no_grad():
        classification_output = classification_model(roi_tensor)
        predicted_class = classification_output.argmax(1).item()
        predicted_class_name = 'Class Name'  # Map class index to class name

    # Draw bounding box and class label on the input image
    plt.imshow(input_image)
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
    plt.text(x1, y1, predicted_class_name, color='red')
    plt.show()