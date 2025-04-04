"""Wrapper classes for various generative models"""

from abc import ABC, abstractmethod
import trimesh
import numpy as np
from tqdm import trange
import torch
from skimage import measure
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from thyroidsynthesis.nn.lightning_module import INRLightningModule
from thyroidsynthesis.nn.data import ThyroidSDFDatamodule
import matplotlib.pyplot as plt


class AbstractModel(ABC):
    def __init__(self, file_path=None):
        if file_path is not None:
            self.load(file_path)

    @abstractmethod
    def fit(self, meshes):
        """Fit model on a list of meshes

        `meshes` should be a list of trimesh meshes
        """
        pass

    @abstractmethod
    def generate_random(self):
        """Generate a random mesh"""
        pass

    @abstractmethod
    def reconstruct(self, mesh):
        """Reconstruct a given mesh with the model"""
        pass

    @abstractmethod
    def load(self, file_path):
        """Load a fitted model from `file_path`"""
        pass

    @abstractmethod
    def save(self, file_path):
        """Save a fitted model to `file_path`"""
        pass


class INR(AbstractModel):
    """Base class for various implicit neural representations"""

    def __init__(self, n_structures, structure_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_structures = n_structures
        self.structure_labels = structure_labels

    def fit(
        self,
        meshes,
        target_meshes=None,
        latent=None,
        epochs=100,
        batch_size=1,
        lr=1e-4,
        clamp_distance=None,
        clamp_start_epoch=0,
        wandb=False,
        enable_checkpointing=False,
        apply_correlation_loss=False,
        apply_batch_correlation_loss=False,
        correlation_loss_prefactor=1,
        apply_latent_loss=False,
        latent_loss_prefactor=1e-2,
        positional_embedding=None,
    ):
        """Fit model on a list of meshes

        `meshes` should be a list of trimesh meshes
        `latent` can be:
            - None -> for fitting on a single mesh, no coditioning
            - list -> fixed latent codes for every mesh in `meshes`
            - int -> trained latent code size
        """
        self.lightning_module = INRLightningModule(
            self.model,
            latent=latent,
            lr=lr,
            clamp_distance=clamp_distance,
            clamp_start_epoch=clamp_start_epoch,
            apply_correlation_loss=apply_correlation_loss,
            apply_batch_correlation_loss=apply_batch_correlation_loss,
            correlation_loss_prefactor=correlation_loss_prefactor,
            apply_latent_loss=apply_latent_loss,
            latent_loss_prefactor=latent_loss_prefactor,
            positional_embedding=positional_embedding,
        )
        dm = ThyroidSDFDatamodule(
            meshes, target_meshes=target_meshes, batch_size=batch_size
        )
        dm.setup()

        if wandb:
            logger = WandbLogger(project="thyroidsynthesis")
        else:
            logger = False

        if enable_checkpointing:
            checkpoint_callback = ModelCheckpoint(
                save_last=True, monitor="train_loss", save_weights_only=True
            )
            callbacks = [checkpoint_callback]
        else:
            callbacks = []

        self.trainer = Trainer(
            max_epochs=epochs,
            logger=logger,
            enable_checkpointing=enable_checkpointing,
            callbacks=callbacks,
        )
        self.trainer.fit(
            model=self.lightning_module,
            datamodule=dm,
        )

    def generate_random(self):
        """Generate a random mesh"""
        resolution = 64
        extent = 1.1
        sampled_sdf = self._sample_sdf(resolution=resolution, extent=extent)
        mesh_dict = {}
        for idx, label in enumerate(self.structure_labels):
            verts, faces, _, _ = measure.marching_cubes(sampled_sdf[idx], 0)
            verts = (2 * extent * verts / (resolution - 1)) - extent
            mesh_dict[label] = trimesh.Trimesh(verts, faces)
            mesh_dict[label].invert()
        return mesh_dict

    def generate_latent(self, latent, resolution=64, extent=1.1):
        """Generate a mesh given a latent code"""
        sampled_sdf = self._sample_sdf(
            resolution=resolution, extent=extent, latent=latent
        )
        mesh_dict = {}
        for idx, label in enumerate(self.structure_labels):
            verts, faces, _, _ = measure.marching_cubes(sampled_sdf[idx], 0)
            verts = (2 * extent * verts / (resolution - 1)) - extent
            mesh_dict[label] = trimesh.Trimesh(verts, faces)
            mesh_dict[label].invert()
        return mesh_dict

    def reconstruct(self, mesh):
        """Reconstruct a given mesh with the model"""
        pass

    def load(self, file_path):
        """Load a fitted model from `file_path`"""
        self.lightning_module = INRLightningModule.load_from_checkpoint(
            file_path, model=self.model
        )

    def save(self, file_path):
        """Save a fitted model to `file_path`"""
        self.trainer.save_checkpoint(file_path)

    def _sample_sdf(
        self, resolution: int, extent: float, latent=None, verbose: bool = False
    ):
        self.lightning_module.eval()

        out_vol = np.zeros((self.n_structures, resolution, resolution, resolution))
        zs = torch.linspace(-extent, extent, steps=out_vol.shape[2])

        for z_it in trange(
            out_vol.shape[2],
            desc="Reconstructing mesh",
            leave=False,
            disable=not verbose,
        ):
            im_slice = np.zeros((resolution, resolution, 1))
            xs = torch.linspace(-extent, extent, steps=im_slice.shape[1])
            ys = torch.linspace(-extent, extent, steps=im_slice.shape[0])

            x, y = torch.meshgrid(xs, ys, indexing="xy")
            z = torch.ones_like(y) * zs[z_it]

            coords = torch.cat(
                [
                    x.reshape((np.prod(im_slice.shape[:2]), 1)),
                    y.reshape((np.prod(im_slice.shape[:2]), 1)),
                    z.reshape((np.prod(im_slice.shape[:2]), 1)),
                ],
                dim=1,
            )
            if torch.cuda.is_available():
                coords = coords.to(device="cuda")

            # Add latent code
            if latent is not None:
                latent_code = torch.tensor(latent).repeat(coords.shape[0], 1)
                coords = torch.cat((coords, latent_code), dim=1).type(coords.dtype)

            if torch.cuda.is_available():
                coords = coords.to(device="cuda")

            with torch.no_grad():
                out, _ = self.lightning_module(coords)

            final_output = np.reshape(
                out.cpu().numpy(), im_slice.shape[:2] + (self.n_structures,)
            ).T
            if z_it == 50 and False:
                cmap = "RdBu"
                plt.subplot(121)
                plt.imshow(
                    final_output[0], cmap=cmap, vmin=-1, vmax=+1, extent=(-1, 1, -1, 1)
                )
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(
                    final_output[0],
                    cmap=cmap,
                    vmin=-0.1,
                    vmax=+0.1,
                    extent=(-1, 1, -1, 1),
                )
                plt.colorbar()
                plt.show()
            out_vol[:, :, :, z_it] = final_output

        return out_vol
