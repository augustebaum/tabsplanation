"""Load training and test data usable in training the classifier and the autoencoder."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import lightning as pl

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler

from tabsplanation.types import D, N, Tensor


class CakeOnSeaDataset(Dataset):
    """A synthetic dataset with 3 classes and only the first 2 features have
    influence on them; the rest are just uniform in $[0, 10]$.

    The number of features to keep (max. 500) can be tuned with the `nb_dims`
    parameters.
    """

    output_dim = 3

    def __init__(
        self,
        xs_path: Path,
        ys_path: Path,
        coefs_path: Path,
        nb_dims: int,
        device: torch.device,
    ):
        """Load the dataset, only keeping the first `nb_dims` columns.

        The columns are normalized.

        Inputs:
        -------
        * `xs_path`: Path where the input data are stored.
        * `ys_path`: Path where the class data are stored.
        * `coefs_path`: Path where the coefficients used to generate the extra
            columns are stored.
        * `nb_dims`: Number of columns to keep (from left to right -- the first
            and second columns being the relevant features for classification).
            If negative or zero, take all columns.
        """

        def load_from(path: Path, dtype) -> Tensor:
            arr: np.array = np.load(path)
            return torch.from_numpy(arr.astype(dtype)).to(device)

        self.X = load_from(xs_path, np.float32)
        if nb_dims > 0:
            self.X = self.X[:, :nb_dims]
        self.normalize = Normalize.new(self.X)
        # Classifier requires that the output be 1-dimensional
        self.y = load_from(ys_path, np.int64).squeeze()

        self.input_dim = self.X.shape[1]

        self.coefs = load_from(coefs_path, np.int8)
        if nb_dims > 0:
            self.coefs = self.coefs[:, : (nb_dims - 2)]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input, target = self.normalize(self.X[idx]), self.y[idx]
        # if self.transform is not None:
        #     input = self.transform(self.X[idx])
        return input, target

    def fill_from_2d_point(self, x: Tensor[N, 2]) -> Tensor[N, D]:
        """Take $x_0$ and $x_1$ (un-normalized) and re-create the other columns using
        the same coefficients that were originally used.
        """
        correlated_columns = x @ self.coefs
        points = torch.hstack([x, correlated_columns])
        return points

    @property
    def normalize_inverse(self):
        return NormalizeInverse.new(self.normalize)


class PandasDataset(Dataset):
    """Dataset from `pandas` `DataFrame`."""

    def __init__(
        self,
        csv_path: Path,
        device: torch.device,
    ):
        """Initialize the dataset.

        Inputs:
        -------
        * csv_path: Path to CSV where last column is true label and all other columns are floats.
        """
        df = pd.read_csv(csv_path)

        X = df.iloc[:, 0:-1].to_numpy()
        self.X = torch.from_numpy(X.astype(np.float32)).to(device)

        y = df.iloc[:, -1].to_numpy()
        self.y = torch.from_numpy(y).to(device)

        self.input_dim = self.X.shape[1]
        self.normalize = Normalize.new(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input, target = self.normalize(self.X[idx]), self.y[idx]
        return input, target

    @property
    def normalize_inverse(self):
        return NormalizeInverse.new(self.normalize)


class ForestCoverDataset(PandasDataset):
    """ForestCover without all the binary columns."""

    output_dim = 7


class WineQualityDataset(PandasDataset):

    output_dim = 3


class OnlineNewsPopularityDataset(PandasDataset):

    output_dim = 3


@dataclass
class CakeOnSeaDataModule(pl.LightningDataModule):

    dataset: Dataset
    validation_data_proportion: float
    test_data_proportion: float
    batch_size: int
    correct_for_class_imbalance: bool

    def __post_init__(self):
        super(CakeOnSeaDataModule, self).__init__()

        self.input_dim = self.dataset.input_dim
        self.output_dim = self.dataset.output_dim

        self.setup("")

    def setup(self, stage: str):
        nb_points = len(self.dataset)

        val_size = int(self.validation_data_proportion * nb_points)
        test_size = int(self.test_data_proportion * nb_points)
        train_size = nb_points - (val_size + test_size)
        assert train_size > 0, "No training points; check input proportions"

        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        if self.correct_for_class_imbalance:
            weights = 1 / np.unique(self.train_set.dataset.y, return_counts=True)[1]
            sampler = WeightedRandomSampler(
                weights=weights, num_samples=len(self.train_set), replacement=True
            )
            return DataLoader(
                self.train_set,
                sampler=sampler,
                drop_last=True,
                batch_size=self.batch_size,
            )
        return DataLoader(
            self.train_set, shuffle=True, drop_last=True, batch_size=self.batch_size
        )

    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self, batch_size=None):
        return DataLoader(
            self.test_set,
            shuffle=False,
            batch_size=self.batch_size if batch_size is None else batch_size,
        )

    def predict_dataloader(self):
        pass

    @property
    def test_data(self):
        return self.test_set.dataset[self.test_set.indices]


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


class Normalize:
    def __init__(self, mean: Tensor[1, D], stddev: Tensor[1, D]):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, tensor: Tensor[N, D]):
        """Normalize `tensor`."""
        # TODO: What if there are zeros in self.stddev?
        # In theory, if that is the case all number are equal to the mean so
        # tensor - mean = 0 so the whole thing is 0.
        return (tensor - self.mean) / self.stddev

    @staticmethod
    def new(tensor: Tensor[N, D]) -> "Normalize":
        """Make a new `Normalize` instance from computing the
        mean and standard deviation of `tensor`."""

        stddev, mean = torch.std_mean(tensor, dim=0, unbiased=True)
        stddev = torch.nan_to_num(stddev)
        return Normalize(mean, stddev)


class NormalizeInverse:
    def __init__(self, mean: Tensor[1, D], stddev: Tensor[1, D]):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, tensor: Tensor[N, D]):
        """Unnormalize `tensor`."""
        return tensor * self.stddev + self.mean

    def new(normalize: Normalize) -> "NormalizeInverse":
        return NormalizeInverse(normalize.mean, normalize.stddev)
