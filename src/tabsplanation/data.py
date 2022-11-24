"""Load training and test data usable in training the classifier and the autoencoder."""

from pathlib import Path
from typing import Optional, Tuple, TypeAlias

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchtyping import TensorType


class SyntheticDataset(Dataset):
    """A synthetic dataset with 3 classes and only the first 2 features have
    influence on them; the rest are just uniform in $[0, 10]$.

    The number of features to keep (max. 500) can be tuned with the `nb_dims`
    parameters.
    """

    def __init__(
        self,
        xs_path: Path,
        ys_path: Path,
        nb_dims: int,
        device: torch.device,
    ):
        """Load the dataset, only keeping the first `nb_dims` columns.

        The columns are normalized.

        Inputs:
        -------
        `nb_dims`: Number of columns to keep (from left to right -- the first
            and second columns being the relevant features for classification).
            If negative or zero, take all columns.
        """

        def load_from(path: Path, dtype) -> torch.Tensor:
            arr = np.load(path)
            return torch.tensor(arr).to(device).to(dtype)

        self.X = load_from(xs_path, torch.float)
        if nb_dims > 0:
            self.X = self.X[:, :nb_dims]
        self.normalize = Normalize.new(self.X)
        # Classifier requires that the output be 1-dimensional
        self.y = load_from(ys_path, torch.long).squeeze()

        self.input_dim = self.X.shape[1]
        self.output_dim = len(np.unique(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        input, target = self.normalize(self.X[idx]), self.y[idx]
        # if self.transform is not None:
        #     input = self.transform(self.X[idx])
        return input, target

    @property
    def normalize_inverse(self):
        return NormalizeInverse.new(self.normalize)


class DiabetesDataset(Dataset):
    """PyTorch-compatible dataset, written for the Pima Diabetes dataset.

    Attributes:
    -----------
    df: DataFrame containing all the unmodified data.
    data: Tensor containing all the unmodified data.
    feature_names: List of the column names (except for the last column).
    X: Tensor containing all the standardized feature data.
    y: Tensor containing all the outcome data.
    """

    def __init__(
        self,
        csv_path: Path,
        device: torch.device,
    ):
        self.df = pd.read_csv(csv_path)
        self.feature_names = list(self.df.columns)[:-1]
        self.data = torch.tensor(self.df.to_numpy()).to(device).to(torch.float)

        X, y = self.data[:, 0:8], self.data[:, 8]
        # Standardize features to avoid gradient issues during training
        self.X, self._mean_X, self._stddev_X = standardize(X)
        self.y = y.to(torch.float).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def input_dim(self):
        return len(self.feature_names)

    def remove_standardisation(self, X: torch.Tensor) -> torch.Tensor:
        """Removes standardization from a (hopefully standardized) tensor of
        feature vectors."""
        return X * self._stddev_X + self._mean_X

    @property
    def stats_X(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the feature-wise mean and standard deviation of the whole
        dataset."""
        return self._mean_X, self._stddev_X


def standardize(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the feature-wise standardized input along with
    the mean and standard deviation of the input.

    Inputs:
    -------
    X: Input tensor to standardize.

    Returns:
    --------
    std_X: Input tensor standardized
    mean_X: Input tensor feature-wise mean
    stddev_X: Input tensor feature-wise (unbiased) standard deviation
    """
    stddev_X, mean_X = torch.std_mean(X, dim=0, unbiased=True)
    std_X = (X - mean_X) / stddev_X
    std_X = torch.nan_to_num(std_X)
    return std_X, mean_X, stddev_X


TrainDataset: TypeAlias = Dataset
ValidationDataset: TypeAlias = Dataset
TestDataset: TypeAlias = Dataset

TrainLoader: TypeAlias = DataLoader
ValidationLoader: TypeAlias = DataLoader
TestLoader: TypeAlias = DataLoader


def make_subsets(
    dataset: Dataset,
    val_data_proportion: float,
    test_data_proportion: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[TrainDataset, ValidationDataset, TestDataset]:
    """Split a `Dataset` into train and test datasets."""
    val_size = int(val_data_proportion * len(dataset))
    test_size = int(test_data_proportion * len(dataset))
    train_size = len(dataset) - (val_size + test_size)
    assert train_size >= 0, "Input proportions account for too much"

    return random_split(dataset, [train_size, val_size, test_size], generator)


def make_dataloaders(
    train_dataset: TrainDataset,
    validation_dataset: ValidationDataset,
    test_dataset: TestDataset,
    weighted_sampler: bool,
    **kwargs
) -> Tuple[TrainLoader, ValidationLoader, TestLoader]:
    """Make dataloaders from `Dataset`s.

    If `weighted_sampler` is set to `True`, class weights will be
    computed on the entire dataset so that
    the sampling can be adjusted to prevent class imbalance.
    """

    if weighted_sampler:
        weights = 1 / np.unique(train_dataset.dataset.y, return_counts=True)[1]
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(train_dataset), replacement=True
        )
        train_loader = DataLoader(train_dataset, sampler=sampler, **kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **kwargs)
    return train_loader, validation_loader, test_loader


DatasetsDict: TypeAlias = dict
LoadersDict: TypeAlias = dict


def split_dataset(
    dataset: Dataset,
    val_data_proportion: float,
    test_data_proportion: float,
    batch_size: int,
    weighted_sampler: bool,
) -> Tuple[DatasetsDict, LoadersDict]:
    train_dataset, val_dataset, test_dataset = make_subsets(
        dataset, val_data_proportion, test_data_proportion
    )
    train_loader, val_loader, test_loader = make_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        weighted_sampler=weighted_sampler,
    )
    datasets = {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    }
    loaders = {"train": train_loader, "validation": val_loader, "test": test_loader}
    return datasets, loaders


class Normalize:
    def __init__(
        self, mean: TensorType[1, "input_dim"], stddev: TensorType[1, "input_dim"]
    ):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, tensor: TensorType["nb_points", "input_dim"]):
        """Normalize `tensor`."""
        # TODO: What if there are zeros in self.stddev?
        # In theory, if that is the case all number are equal to the mean so
        # tensor - mean = 0 so the whole thing is 0.
        return (tensor - self.mean) / self.stddev

    @staticmethod
    def new(tensor: TensorType["nb_points", "input_dim"]) -> "Normalize":
        """Make a new `Normalize` instance from computing the
        mean and standard deviation of `tensor`."""
        stddev, mean = torch.std_mean(tensor, dim=0, unbiased=True)
        stddev = torch.nan_to_num(stddev)
        return Normalize(mean, stddev)


class NormalizeInverse:
    def __init__(
        self, mean: TensorType[1, "input_dim"], stddev: TensorType[1, "input_dim"]
    ):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, tensor: TensorType["nb_points", "input_dim"]):
        """Unnormalize `tensor`."""
        return tensor * self.stddev + self.mean

    def new(normalize: Normalize) -> "NormalizeInverse":
        return NormalizeInverse(normalize.mean, normalize.stddev)
