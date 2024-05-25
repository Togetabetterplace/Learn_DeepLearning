import time
import torch.utils.data
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torchvision.transforms
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch


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


def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_model(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    # net.load_state_dict(torch.load(r'model\ResModel38_old.pth'))
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.09)
    loss = nn.CrossEntropyLoss()
    timer1, num_batches = time.time(), len(train_iter)
    best_test_acc = 0

    # Initialize lists to store training and testing metrics
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        # 积累器
        net.train()  # 设定训练模式
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        # print(f'test acc {test_acc:.3f}')
        if test_acc > best_test_acc:
            # 保存模型
            torch.save(net.state_dict(), 'model/Model'+str(epoch)+'.pth')
            print(f"save Model in 'model/Model{str(epoch)}.pth'...")
            best_test_acc = test_acc
            print(f'loss {train_l:.3f}, train acc {train_acc:.3f}', f'test acc {test_acc:.3f}')


    # Plot the metrics
    plt.figure()
    # 导入配色

    plt.plot(range(num_epochs), train_loss_list, 'co', label='Train Loss')
    plt.plot(range(num_epochs), train_acc_list, 'b-', label='Train Acc')
    plt.plot(range(num_epochs), test_acc_list, 'r-', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.show()

    # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
    #     animator.add(epoch + (i + 1) / num_batches,
    #                  (train_l, train_acc, None))
    timer2 = time.time()
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{timer1-timer2:.1f} examples/sec '
          f'on {str(device)}')
