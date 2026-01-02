import torch
from torch import optim
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
from torchvision import datasets  # Used for loading in pre-made datasets
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn.functional as F

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))


class CNNModel(nn.Module):
    """
    Defining CNN by inheriting nn.Module for functions and framework
    -  Feature extraction: 2 convolutional layers
    - Downscaling: 1 pooling layer
    - Classification: 2 fully connected and dense layers

    Height and width are pixels ofc. fashionMNIST is grayscale [batch_size,1,28,28]
    """
    # init will run when an instance of CNNModel is made
    def __init__(self):
        # Running it while inheriting from parent nn.Module
        super(CNNModel, self).__init__()
        # 1 input channel (grayscale), 32 filters [3x3] for 32 features, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # Now taking the 32 feature maps [input] and applying 64 new filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # Taking a 2,2 window and keeping the max size. It covers 2x2 region,
        # jumping 2 pixels at a time and keeps only the single largest value from that block.
        # It's like when applying a max filter
        self.pool = nn.MaxPool2d(2, 2)
        # There is a formula applied everytime it is convoluted, giving its spatial size [HxW]
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Flattened input size of 64 feature maps * 5 height * 5 width, return 128 neurons (condensing the 1600)
        self.fc2 = nn.Linear(128, 10) # 10 output neurons, each corresponds to a class (FashionMNIST)

    # Auto-ran by PyTorch when instantiated model is called again e.g. output=model(images)
    # Literally same as output = model.forward(images)
    def forward(self, x):
        """
        Defines the forward pass of the CNN model.

        Given an input tensor 'x' (a batch of images), the data flows through
        the network in the following sequence:

        1. conv1 -> ReLU -> pool:
           - The first convolution extracts 32 low-level features (edges, textures, etc.)
           - ReLU removes negative activations to introduce non-linearity
           - Max pooling reduces spatial dimensions (downsampling)

        2. conv2 -> ReLU -> pool:
           - The second convolution extracts 64 higher-level features
           - Again followed by ReLU and pooling to compress the feature maps

        3. flatten:
           - Converts the 3D feature maps into a 2D tensor (batch_size, features)
           - e.g. [64, 64, 5, 5] → [64, 1600]

        4. fc1 -> ReLU:
           - Fully connected layer that maps 1600 raw features into 128 learned representations

        5. fc2:
           - Final fully connected layer that outputs 10 class scores (logits)
             corresponding to the 10 FashionMNIST categories

        Returns:
            x: Tensor of shape [batch_size, 10] containing the raw class scores (logits)
        """
        x = self.pool(F.relu(self.conv1(x)))  # conv1 + ReLU + pool
        x = self.pool(F.relu(self.conv2(x)))  # conv2 + ReLU + pool
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))  # fully connected layer
        x = self.fc2(x)  # output layer
        # Where x are the logits [raw score] for each class
        # Higher value = more accurate classification
        # A formula is applied to extract probabilities of correct classification
        return x


def imshow(img):
    """
    A function that will turn a tensor argument into a NumPy array,
    format it correctly and uses matplot to display it
    """
    npimg = img.numpy()
    #  Rearranges dimensions of array to be (H,W,C) for matplot
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# show images
#imshow(torchvision.utils.make_grid(train_features))


# Basic Workflow
# Step 1: instantiate the model
# model = CNNModel()
#
# # Step 2: get a batch of images from the dataloader
# images, labels = next(iter(train_dataloader))
#
# # Step 3: run the forward pass
# outputs = model(images)   # internally calls CNNModel.forward(images)
#
# print(outputs.shape)      # torch.Size([64, 10])  ← 64 images, 10 class scores each


# Instantiating the model
model = CNNModel()
criterion = nn.CrossEntropyLoss()          # standard loss function
# Optimizer - An algorithm that updates the model's weights based on loss
# lr = learning rate, how much of a leap it takes in changing weights
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5  # start small for a toy model
epoch_losses = []

for epoch in range(epochs):
    """
 iterating over the dataset multiple times (epochs), and for each small batch of images:

 - Running the model (forward pass)

 - Measuring how wrong it is (loss)

 - Backpropagating the error (backward pass)

 - Updating the model’s weights (optimizer step)

 This cycle is how the  model learns.
    """
    running_loss = 0.0
    for images, labels in train_dataloader:
        # 1. Zero the gradients as we don't want to accumulate from prior batches
        optimizer.zero_grad()

        # 2. Forward pass, basically training on the dataset
        outputs = model(images)

        # 3. Compute loss (ratio of misclassifications)
        loss = criterion(outputs, labels)

        # 4. Backward pass - tells optimiser how to adjust weights to reduce loss next time
        loss.backward()

        # 5. Uses the gradients calculated by backward pass, update the weights a tiny bit [lr=0.001]
        optimizer.step()

        # Accumulate batch loss
        running_loss += loss.item()

    # Compute average loss per epoch
    avg_loss = running_loss / len(train_dataloader)
    epoch_losses.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}")

# Plot loss vs epoch
plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.show()

# Testing
# No gradient as we aren't optimising the model, just testing it
with torch.no_grad():
    test_images, test_labels = next(iter(train_dataloader)) #one batch
    outputs = model(test_images) #batch_size [64], num_classes [10]
    # _ is a throwaway value that contains the max logit value, we just want their labels [predicted]
    _, predicted = torch.max(outputs, 1)
    #“For each row (image), find the column (class) that has the highest logit. This will be
    # the image's class, as each image only has one class”

# prints the first ten predictions and their actual labels to compare
print("Predicted:", predicted[:10])
print("Actual:   ", test_labels[:10])


"""
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
"""
