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
    cmp = y_hat.type(y.dtype) == y  # ����ת��
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
    """������ָ�����ݼ���ģ�͵ľ���"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # ��ģ������Ϊ����ģʽ
    metric = Accumulator(2)  # ��ȷԤ������Ԥ������
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


"""
�������Ƕ�������������ĺ����� �������������������ڲ�������������GPU����������д��롣
"""


def try_gpu(i=0):  # @save
    """������ڣ��򷵻�gpu(i)�����򷵻�cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  # @save
    """�������п��õ�GPU�����û��GPU���򷵻�[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


try_gpu(), try_gpu(10), try_all_gpus()
