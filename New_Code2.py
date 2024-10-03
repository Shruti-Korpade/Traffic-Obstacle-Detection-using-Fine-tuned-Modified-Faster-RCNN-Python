import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

def main():
    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rest of your code remains the same...
    # Define data transformations for the test dataset
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Specify the path to the test dataset directory
    test_data_dir = 'C:/Users/kskor/PycharmProjects/EDI_code/Images/dataset 2'

    # Create a dataset and dataloader for testing
    test_dataset = datasets.ImageFolder(os.path.join(test_data_dir, 'test'), transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load the pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)
    num_classes = 5
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Experiment with different learning rates and optimizers
    learning_rates = [0.001, 0.0001]  # You can add more learning rates to test
    optimizers = [optim.SGD, optim.Adam]  # You can try different optimizers

    for lr in learning_rates:
        for optimizer_class in optimizers:
            # Load the trained model weights for each configuration
            model_name = f'resnet50_model_lr_{lr}{optimizer_class.name_}.pth'
            model.load_state_dict(torch.load(model_name))
            model = model.to(device)
            model.eval()  # Set the model to evaluation mode

            # Define data transformations for the test dataset
            data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # Specify the path to the test dataset directory
            test_data_dir = 'C:/Users/kskor/PycharmProjects/EDI_code/Images/dataset 2'

            # Create a dataset and dataloader for testing
            test_dataset = datasets.ImageFolder(os.path.join(test_data_dir, 'test'), transform=data_transforms)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

            # Testing the trained model
            test_corrects = 0

            with torch.no_grad():
                # Testing loop (assuming you have a test_loader defined)
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    test_corrects += torch.sum(preds == labels.data)

            test_acc = 100 * test_corrects.double() / len(test_dataset)
            print(f'LR: {lr} | Optimizer: {optimizer_class._name_} | Test Accuracy: {test_acc:.4f}%')


if __name__ == '_main_':
    # Call the main function
    main()