#import torch
#from torch.utils.data import Dataset - Used for building your own datasets
from torch.utils.data import DataLoader
from torchvision import datasets  # Used for loading in pre-made datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt

# Loading in dataset (typically as images)
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    # Transforming images to PyTorch tensors
    # Tensors are a PyTorch friendly format, consisting of normalised pixel values (0-1)
    # It's a multi-dimensional array [channels, height, width] representing all pixel values
    # Consider further transformation arguments for complex CNNs
    transform=ToTensor()
)
# Groups data in mini batches of 64, but shuffles the order of samples before grouping
# it's an iterable you can loop through, 1 loop = 64 images each with a label
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
# train_features is the batch of 64 grayscale images
# train_labels is the integer labels (0-9) for each image
print(f"Feature batch shape: {train_features.size()}")
# should yield [64,1,28,28]
# 64 is 64 images in batch
# 1 is number of channels(1=grayscale) (3=rgb)
# 28 is image height in pixels
# 28 is image width in pixels
print(f"Labels batch shape: {train_labels.size()}")

# grabbing the first label
label = train_labels[0]
# "squeeze" removes any dimension of size 1
img = train_features[0].squeeze()
#grayscale color map of the first item in the first batch of training dataset
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

#print(training_data.classes)
# ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
# 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

#print(len(training_data))
#for i in training_data:
#    print(i)
