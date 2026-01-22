import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class MNISTLoader:
    def __init__(self, data_dir='./Datasets', train_split=0.7, retain_split=0.9, batch_size=32):
        """
        Load MNIST dataset and create PyTorch DataLoaders

        Args:
            data_dir: directory to download/load MNIST data
            train_split: proportion of data for training (rest is test)
            retain_split: proportion of train data to retain (rest is forget set)
            batch_size: batch size for DataLoader

        Returns:
            train_loader, test_loader, retain_loader, forget_loader
        """
        self.data_dir = data_dir
        self.train_split = train_split
        self.retain_split = retain_split
        self.batch_size = batch_size

        # Define transforms for MNIST
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])

    def load_data(self):
        """Load MNIST and create train/test/retain/forget splits"""

        # Load full MNIST dataset
        full_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )

        # Load test set separately
        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )

        # Split training data into train and validation
        train_size = int(self.train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, _ = random_split(full_dataset, [train_size, val_size])

        # Split train into retain and forget
        retain_size = int(self.retain_split * len(train_dataset))
        forget_size = len(train_dataset) - retain_size
        retain_dataset, forget_dataset = random_split(train_dataset, [retain_size, forget_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        retain_loader = DataLoader(retain_dataset, batch_size=self.batch_size, shuffle=True)
        forget_loader = DataLoader(forget_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader, retain_loader, forget_loader
