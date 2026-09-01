import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import torchio as tio
import monai.transforms as mt
import monai.transforms as mt
import monai.transforms as mt


class NIfTIFolderDataset(Dataset):
    def __init__(self, root_dir, target_shape=(128, 128, 128), transform=None):
        self.root_dir     = Path(root_dir)
        self.target_shape = target_shape
        self.transform    = transform

        self.CLASSES   = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.label_map = {cls: i for i, cls in enumerate(self.CLASSES)}

        self.samples = []
        for cls in self.CLASSES:
            for ext in ("*.nii", "*.nii.gz"):
                for fpath in (self.root_dir / cls).glob(ext):
                    self.samples.append((fpath, self.label_map[cls]))

        print(f"Found {len(self.samples)} images | Classes: {self.label_map}")

    def __len__(self):
        return len(self.samples)

    def _resize(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            tensor.unsqueeze(0),       # (1, 1, D, H, W)
            size=self.target_shape,
            mode="trilinear",
            align_corners=False
        ).squeeze(0)                   # (1, D, H, W)

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std  = tensor.std()
        return (tensor - mean) / (std + 1e-8)

    def __getitem__(self, idx):
        fpath, label_idx = self.samples[idx]

        img_sitk   = sitk.ReadImage(str(fpath), sitk.sitkFloat32)
        arr        = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        img_tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, D, H, W)

        img_tensor = self._resize(img_tensor)
        img_tensor = self._normalize(img_tensor)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.tensor(label_idx, dtype=torch.long)



class ADNIDataset(Dataset):
    def __init__(self, dataframe, CLASSES, transform=None):
        self.df = dataframe
        self.transform = transform
        self.CLASSES = CLASSES
        # Create map from the provided list to ensure consistent indexing
        self.label_map = {label: i for i, label in enumerate(self.CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load processed NIfTI
        img_sitk = sitk.ReadImage(row['processed_path'], sitk.sitkFloat32)
        img_array = sitk.GetArrayFromImage(img_sitk).astype('float32')
        
        # Convert to tensor and add Channel Dim: (1, D, H, W)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        # Get Label using the fixed label_map
        label_name = row['Group']
        label = torch.tensor(self.label_map[label_name], dtype=torch.long)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # normalize it to min and max
        # img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
            
        return img_tensor, label



class ADNIMultiDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tabular_features: np.ndarray,
        classes: list[str],
        paths: pd.Series,
        transform=None,
    ):
        """
        Multimodal ADNI dataset returning a 3D MRI tensor and a tabular
        feature vector for each subject visit.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Split dataframe (train / val / test) containing at minimum:
            - 'Group' : diagnosis label string
            Index must be reset and aligned with tabular_features rows.
        tabular_features : np.ndarray, shape (N, F)
            Preprocessed tabular feature matrix from preprocess_adni(),
            already imputed, encoded, and scaled.
            Row i must correspond to dataframe.iloc[i].
        classes : list[str]
            Ordered class list e.g. ["AD", "CN", "LMCI"].
            Determines integer label assignment.
        paths : pd.Series, shape (N,)
            Aligned Series of NIfTI file paths from preprocess_adni()
            e.g. data["train_paths"]. Row i must correspond to dataframe.iloc[i].
        transform : callable | None
            Optional volumetric augmentation applied to the MRI tensor
            before normalisation.
        """
        assert len(dataframe) == len(tabular_features) == len(paths), (
            f"dataframe ({len(dataframe)}), tabular_features ({len(tabular_features)}), "
            f"and paths ({len(paths)}) must all have the same length."
        )

        self.df        = dataframe.reset_index(drop=True)
        self.tabular   = tabular_features.astype(np.float32)
        self.paths     = paths.reset_index(drop=True)
        self.transform = transform
        self.classes   = classes
        self.label_map = {label: i for i, label in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        img_tensor : torch.Tensor, shape (1, D, H, W), float32
            Normalised MRI volume.
        tab_tensor : torch.Tensor, shape (F,), float32
            Preprocessed tabular feature vector.
        label      : torch.Tensor, scalar, int64
            Integer class index.
        """
        row = self.df.iloc[idx]

        # ── MRI volume ────────────────────────────────────────────────────────
        # Path comes from the aligned paths Series, not the dataframe
        img_sitk  = sitk.ReadImage(str(self.paths.iloc[idx]), sitk.sitkFloat32)
        img_array = sitk.GetArrayFromImage(img_sitk).astype(np.float32)

        # Add channel dim → (1, D, H, W)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        # Optional volumetric augmentation (applied before normalisation)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Min-max normalise per volume to [0, 1]
        # img_tensor = (img_tensor - img_tensor.min()) / (
        #     img_tensor.max() - img_tensor.min() + 1e-8
        # )

        # ── Tabular features ──────────────────────────────────────────────────
        tab_tensor = torch.from_numpy(self.tabular[idx])   # shape: (F,)

        # ── Label ─────────────────────────────────────────────────────────────
        label = torch.tensor(
            self.label_map[row["Group"]], dtype=torch.long
        )

        return img_tensor, tab_tensor, label



class ADNITabularDataset(Dataset):
    """
    Tabular-only ADNI dataset returning a feature vector and class label
    for each subject visit.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Split dataframe (train / val / test) containing at minimum:
        - 'Group' : diagnosis label string (e.g. "AD", "CN", "LMCI")
        Index must be reset and aligned with tabular_features rows.
    tabular_features : np.ndarray, shape (N, F)
        Preprocessed tabular feature matrix from preprocess_adni(),
        already imputed, encoded, and scaled.
        Row i must correspond to dataframe.iloc[i].
    classes : list[str]
        Ordered class list e.g. ["AD", "CN", "LMCI"].
        Determines integer label assignment (position = class index).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tabular_features: np.ndarray,
        classes: list[str],
    ):
        assert len(dataframe) == len(tabular_features), (
            f"dataframe ({len(dataframe)}) and tabular_features "
            f"({len(tabular_features)}) must have the same length."
        )

        self.df        = dataframe.reset_index(drop=True)
        self.tabular   = tabular_features.astype(np.float32)
        self.classes   = classes
        self.label_map = {label: i for i, label in enumerate(classes)}

    # ------------------------------------------------------------------
    # Required Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        tab_tensor : torch.Tensor, shape (F,), float32
            Preprocessed tabular feature vector.
        label      : torch.Tensor, scalar, int64
            Integer class index.
        """
        row        = self.df.iloc[idx]
        tab_tensor = torch.from_numpy(self.tabular[idx])          # (F,)
        label      = torch.tensor(
            self.label_map[row["Group"]], dtype=torch.long
        )
        return tab_tensor, label

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        """Number of tabular features — pass directly to TabularClassifier."""
        return self.tabular.shape[1]

    @property
    def n_classes(self) -> int:
        """Number of classes — pass directly to TabularClassifier."""
        return len(self.classes)

    def class_counts(self) -> dict[str, int]:
        """Raw per-class sample counts (useful for building loss weights)."""
        return self.df["Group"].value_counts().to_dict()

    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights aligned to self.classes order.
        Ready to pass straight to nn.CrossEntropyLoss(weight=...).

        Example
        -------
        criterion = nn.CrossEntropyLoss(weight=train_ds.class_weights().to(device))
        """
        counts  = self.df["Group"].value_counts()
        weights = torch.tensor(
            [1.0 / counts[cls] for cls in self.classes], dtype=torch.float
        )
        return weights / weights.sum()

if __name__=="__main__":
    pass
