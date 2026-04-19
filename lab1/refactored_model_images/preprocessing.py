from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch

def get_train_loader(batch_size, size = 100, transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])) -> DataLoader:
    _transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

    _train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    _train_subset = Subset(_train_dataset, range(size))
    return DataLoader(_train_subset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(42))

def get_test_loader(batch_size, size = 30, transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
  ) -> DataLoader:
    _test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    _test_subset = Subset(_test_dataset, range(size))
    return DataLoader(_test_subset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(42))