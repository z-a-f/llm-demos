import functools
import os

import torch

import torchvision as tv
import torchvision.transforms.v2  # Need to import this to init

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

from .utils import compute_mean_std

dataset_classes = {
    'MNIST': tv.datasets.MNIST,
    'EMNIST': functools.partial(tv.datasets.EMNIST, split='mnist')
}


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset: str,
                 data_dir: str,
                 batch_size: int = 32,
                 seed: int = 42,
                 mean_std: tuple = None,  # Mean and Std
                ):
        super().__init__()
        self._dataset_cls = None  # This is the callable that will be used to create the dataset
        self._dataset_name = None

        self.dataset = dataset
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed

        self.transform = tv.transforms.v2.Compose([
            tv.transforms.v2.ToImage(),
            # tv.transforms.v2.ToDtype(torch.uint8, scale=True),  # Only use if there are image transformation later
            tv.transforms.v2.ToDtype(torch.float32, scale=True),   # Normalize expects float input
            # Normalization parameters will be added later
            # tv.transforms.v2.Normalize(mean=1.0, std=1.0),
        ])

        self._mean_std = None
        self.mean_std = mean_std

    @property
    def mean_std(self):
        return self._mean_std

    @mean_std.setter
    def mean_std(self, ms: tuple):
        if ms == self._mean_std:
            return
        self._mean_std = ms
        # Find the normalization layer in the transorms
        tr = None
        for idx, tr in enumerate(self.transform.transforms):
            if isinstance(tr, (tv.transforms.v2.Normalize, tv.transforms.Normalize)):
                tr_idx = idx  # For scope
                break
        else:
            tr = tv.transforms.v2.Normalize(mean=[1.0], std=[1.0])
            self.transform.transforms.append(tr)
            tr_idx = len(self.transform.transforms) - 1

        if ms is None:  # Remove the normalization
            del self.transform.transforms[tr_idx]
            return

        tr.mean = ms[0]
        tr.std = ms[1]

    @property
    def dataset(self):
        if self._dataset_cls is None:
            return None
        return self._dataset_name

    @dataset.setter
    def dataset(self, name: str):
        name = name.upper()
        self._dataset_cls = dataset_classes[name]
        self._dataset_name = name

    def prepare_data(self):
        # Download
        train_set = self._dataset_cls(self.data_dir, train=True, download=True)
        self._dataset_cls(self.data_dir, train=False, download=True)
        # If the normalization was not provided, compute it here
        if self._mean_std is None:
            self.mean_std = compute_mean_std(train_set)
        return self

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = self._dataset_cls(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(self.seed)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = self._dataset_cls(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = self._dataset_cls(self.data_dir, train=False, transform=self.transform)

        return self

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_train,
                                           batch_size=self.batch_size,
                                           persistent_workers=True,
                                           shuffle=True,
                                           num_workers=os.cpu_count())

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_val,
                                           batch_size=self.batch_size,
                                           persistent_workers=True,
                                           # shuffle=True,
                                           num_workers=os.cpu_count(),)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test,
                                           batch_size=self.batch_size,
                                           persistent_workers=False,
                                           num_workers=os.cpu_count())

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_predict,
                                           batch_size=self.batch_size,
                                           persistent_workers=False,
                                           num_workers=os.cpu_count())

    # def teardown(self, stage: str):
    #     # Used to clean-up when the run is finished
    #     pass