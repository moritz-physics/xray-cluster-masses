import numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomCrop, Resizing

class Preprocessor:
    """Holds an image cube + parallel feature arrays, with normalisation, splitting and augmentation helpers."""

    def __init__(self, images, features: dict):
        """Wrap an image cube (shape (N, H, W, C)) and a parallel dict of per-cluster feature arrays."""
        self.images = np.array(images, dtype=np.float32)     # shape: (N, H, W, C) — clusters, height, width, channels
        self.features = {
            name: np.array(val, dtype=np.float32) for name, val in features.items()
        }
        self.feature_stats = {}

    def get_feature(self, name:str, as_column=False):
        """ Retrieve a feature BY name.

        Parameters
        ----------
        name : str
            Name of the feature.
        as_column : bool, optional
            If True, returns as column vector (N, 1). 
            Otherwise, returns as 1D array.

        Returns
        -------
        np.ndarray
            Feature array.
        """
        arr = self.features[name]
        if as_column and arr.ndim == 1:
            return arr.reshape(-1,1)
        return arr

    def normalize_feature(self, name: str):
        """Normalize a feature to zero mean and unit variance.

        Parameters
        ----------
        name : str
            Name of the feature to normalize.

        Returns
        -------
        np.ndarray
            Normalized feature array.
        """
        feat = self.get_feature(name)
        mean = np.mean(feat)
        std = np.std(feat)

        self.features[name] = (feat - mean) / (std + 1e-8)
        self.feature_stats[name] = {"mean": mean, "std": std}
        return self.features[name]
    

    def denormalize_feature(self, name: str, arr_norm):
        """Convert normalized feature values back to original units.

        Parameters
        ----------
        name : str
            Feature name.
        arr_norm : float or np.ndarray
            Normalized value(s).

        Returns
        -------
        float or np.ndarray
            Denormalized value(s).
        """
        stats = self.feature_stats[name]
        mean = stats["mean"]
        std = stats["std"]
        return arr_norm * (std + 1e-8) + mean

    def normalize_images(self, mode: str='per_image'):
        """
        Normalize images either per image or per channel.

        Parameters
        ----------
        mode : str, optional
            'per_image' or 'per_channel' normalization mode.

        Returns
        -------
        np.ndarray
            Normalized image array.
        """

        match mode:
            case 'per_image':
                img_mean = self.images.mean(axis=(1, 2, 3), keepdims=True)
                img_std = self.images.std(axis=(1, 2, 3), keepdims=True)

            case 'per_channel':
                img_mean = self.images.mean(axis=(1, 2), keepdims=True)  # shape: (N, 1, 1, C)
                img_std = self.images.std(axis=(1, 2), keepdims=True)

            case _:
                raise ValueError("Invalid mode. Use 'per_image' or 'per_channel'.")

        self.images = (self.images - img_mean) / (img_std + 1e-8)
        return self.images


    def split(self, test_size=0.15, val_size=0.15, random_state=42):
        """Split dataset into train, validation, and test sets.

        Parameters
        ----------
        test_size : float, optional
            Fraction for test set.
        val_size : float, optional
            Fraction for validation set.
        random_state : int, optional
            Random seed.

        Returns
        -------
        tuple
            (train, val, test) as Preprocessor instances.
        """
        n_samples = len(self.images)
        indices = np.arange(n_samples)

        # Split  test set
        train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

        # Split train into train + val
        val_relative_size = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_relative_size, random_state=random_state)

        # Create new Preprocessor instances
        def new_inst(idx):
            return Preprocessor(
                images=self.images[idx],
                features={name: val[idx] for name, val in self.features.items()}
            )

        return new_inst(train_idx), new_inst(val_idx), new_inst(test_idx)


    def get_xy(self, target_feature_name: str):
        """Get input images and a target feature array for training.

        Parameters
        ----------
        target_feature_name : str
            Name of target feature (e.g., 'mass').

        Returns
        -------
        tuple
            (images, target_feature) arrays.
        """

        return self.images, self.get_feature(target_feature_name)


    def augment_images(self, batch_size=128):
        """
        Perform random image augmentations in batches.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for augmentation.

        Returns
        -------
        np.ndarray
            Augmented images.
        """
        keras_augmentor = Sequential([
            RandomFlip("horizontal_and_vertical"),
            RandomRotation(0.1),
            RandomZoom(0.1),
            RandomCrop(45, 45),       # crop+resize was tested and slightly degraded results — kept for completeness
            Resizing(50, 50),         # resize back to the original 50x50 shape
        ])

        augmented = []
        for i in range(0, len(self.images), batch_size):
            batch = self.images[i:i+batch_size]
            aug_batch = keras_augmentor(batch, training=True).numpy()
            augmented.append(aug_batch)
        
        self.images = np.concatenate(augmented, axis=0)
        return self.images

