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
    cmp = y_hat.type(y.dtype) == y  # 类型转换
    return float(cmp.type(y.dtype).sum())

# accuracy(y_hat, y) / len(y)


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


"""
现在我们定义了两个方便的函数， 这两个函数允许我们在不存在所需所有GPU的情况下运行代码。
"""


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  # @save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
