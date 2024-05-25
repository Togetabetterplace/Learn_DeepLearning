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
