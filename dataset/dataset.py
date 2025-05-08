import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import uniform_filter

def compute_spectral_variation(data, window_size=3):
    """
    Computes the spectral variation image using a mean filter, which is equivalent to smoothing.
    Args:
        data (np.ndarray): Hyperspectral data with shape (channels, width, height).
        window_size (int): Size of the filter window.
    Returns:
        np.ndarray: Smoothed image data.
    """
    channels, width, height = data.shape
    smoothed_data = np.zeros_like(data)

    for i in range(channels):
        smoothed_data[i] = uniform_filter(data[i], size=window_size)  # Apply a uniform filter (mean filter)

    return smoothed_data

class Patch_data_loader(Dataset):
    """
    Data loader for extracting patches from the hyperspectral data.
    """
    def __init__(self, data, neighbor, overlap):
        """
        Initializes the Patch_data_loader.

        Args:
            data (np.ndarray): Hyperspectral data with shape (channels, width, height).
            neighbor (int): Size of the neighbor patch (must be odd).
            overlap (int): Number of overlapping pixels between adjacent patches.
        """
        super(Patch_data_loader, self).__init__()

        self.all_data = data
        self.neighbor = neighbor
        self.cube_data = []  # List to store the extracted neighbor patches (9x9 in the original comment)
        [_, w, h] = self.all_data.shape
        self.stride = self.neighbor - overlap
        for i in range(0, w - (self.neighbor - 1), self.stride):
            for j in range(0, h - (self.neighbor - 1), self.stride):
                self.cube_data.append(self.all_data[:, i:i + self.neighbor, j:j + self.neighbor])
        self.cube_data = np.array(self.cube_data, dtype='float32')


    def __getitem__(self, index):
        """
        Retrieves a patch from the dataset based on the index.

        Args:
            index (int): Index of the patch to retrieve.

        Returns:
            torch.Tensor: The extracted patch as a PyTorch tensor (channels, neighbor, neighbor).
        """
        cube_data = torch.tensor(self.cube_data[index, :, :, :], dtype=torch.float32)

        return cube_data

    def __len__(self):
        """
        Returns the total number of patches in the dataset.

        Returns:
            int: Total number of patches.
        """
        return len(self.cube_data)


class Patch_data_loader_train(Dataset):
    """
    Data loader for extracting and randomly selecting patches for training, along with their smoothed versions.
    """
    def __init__(self, data, neighbor, overlap, ratio):
        """
        Initializes the Patch_data_loader_train.

        Args:
            data (np.ndarray): Hyperspectral data with shape (channels, width, height).
            neighbor (int): Size of the neighbor patch (must be odd).
            overlap (int): Number of overlapping pixels between adjacent patches.
            ratio (float): Fraction of total patches to be selected for training.
        """
        super(Patch_data_loader_train, self).__init__()

        self.all_data = data
        self.smoothed_data = compute_spectral_variation(data, window_size=3)
        self.neighbor = neighbor
        self.cube_data = []  # List to store the extracted neighbor patches (9x9 in the original comment)
        self.var_cube_data = []  # List to store the smoothed neighbor patches (9x9 in the original comment)
        [_, w, h] = self.all_data.shape
        self.stride = self.neighbor - overlap

        # Collect all patches
        for i in range(0, w - (self.neighbor - 1), self.stride):
            for j in range(0, h - (self.neighbor - 1), self.stride):
                self.cube_data.append(self.all_data[:, i:i + self.neighbor, j:j + self.neighbor])
                self.var_cube_data.append(self.smoothed_data[:, i:i + self.neighbor, j:j + self.neighbor])

        self.cube_data = np.array(self.cube_data, dtype='float32')
        self.var_cube_data = np.array(self.var_cube_data, dtype='float32')

        # Randomly select patches according to the ratio
        total_patches = len(self.cube_data)
        selected_count = int(total_patches * ratio)
        self.selected_indices = np.random.choice(total_patches, size=selected_count, replace=False)  # Randomly select indices

    def __getitem__(self, index):
        """
        Retrieves a randomly selected patch and its smoothed version based on the index.

        Args:
            index (int): Index of the selected patch to retrieve.

        Returns:
            tuple: A tuple containing the original patch (torch.Tensor) and its smoothed version (torch.Tensor),
                   both with shape (channels, neighbor, neighbor).
        """
        cube_data = torch.tensor(self.cube_data[self.selected_indices[index], :, :, :], dtype=torch.float32)

        var_cube_data = torch.tensor(self.var_cube_data[self.selected_indices[index], :, :, :], dtype=torch.float32)

        return cube_data, var_cube_data

    def __len__(self):
        """
        Returns the number of selected patches for training.

        Returns:
            int: Number of selected patches.
        """
        return len(self.selected_indices)