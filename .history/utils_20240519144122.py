import torch.utils.data
from IPython import display
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torchvision.transforms
import numpy as np


def load_data_fashion_mnist(batch_size, resize=0):
    """Download the Fashion-MNIST dataset and then load into memory."""
    train_dataset = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)  # type: ignore

    test_dataset = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)  # type: ignore

    if resize != 0:
        resize_fn = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor()
        ])
        train_dataset.transform = resize_fn
        test_dataset.transform = resize_fn
    return train_loader, test_loader


def accuracy(y_hat, y):
    """
    Compute the number of correct predictions.

    Parameters:
        y_hat (torch.Tensor): A tensor of predicted probabilities with shape
            (batch_size, num_classes)
        y (torch.Tensor): A tensor of true labels with shape (batch_size,)

    Returns:
        float: The number of correct predictions.

    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # ÀàĞÍ×ª»»
    return float(cmp.type(y.dtype).sum())

# accuracy(y_hat, y) / len(y)
