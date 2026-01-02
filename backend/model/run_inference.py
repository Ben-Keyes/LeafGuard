import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from backend.model.CNN import CNNModel  # importing model class

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

if __name__ == '__main__':

    print("Running testing...")
    # Loading the test dataset again
    test_dataset = datasets.ImageFolder(
        r"C:\datasets\PlantVillage\unaugmented",
        transform=transform
    )
    # Batch size 32, no workers is faster
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Loading the model
    num_classes = len(test_dataset.classes)
    model = CNNModel(num_classes)
    model.load_state_dict(torch.load("leaf_5_epoch_cnn.pth"))
    model.eval()

    # Test loop
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Full Test Accuracy: {100 * correct / total:.2f}%")
