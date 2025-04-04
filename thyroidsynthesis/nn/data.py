"""Dataset and datamodule for presampled SDF data"""

from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
import trimesh
import numpy as np


class ThyroidPresampledSDFDataset(Dataset):
    """Dataset which loads samples of presampled SDF."""

    def __init__(self, file_paths, debug=False):
        self.points = []
        self.sdf = []
        for file_path in file_paths:
            data = np.load(file_path)
            self.points.append(data["points"])
            self.sdf.append(data["sdf"])
        self.n_samples, self.n_structures = self.sdf[0].shape

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx: int):
        n_points = 1000
        indices = np.random.choice(self.n_samples, n_points)

        points = self.points[idx][indices]
        sdf = self.sdf[idx][indices]

        return idx, points.astype(np.float32), sdf.astype(np.float32)


class ThyroidSDFDatamodule(LightningDataModule):
    def __init__(
        self,
        meshes,
        target_meshes=None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["meshes", "target_meshes"])
        self.meshes = meshes
        self.dataset_class = ThyroidPresampledSDFDataset

        # Handle target meshes
        self.target_meshes = target_meshes

        # Convert to dict format if only single meshes are given
        if target_meshes is not None:
            if type(self.target_meshes[0]) is trimesh.base.Trimesh:
                self.target_meshes = [{"structure": m} for m in self.target_meshes]

    def setup(self, stage=None) -> None:
        self.data_train = self.dataset_class(self.meshes)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
