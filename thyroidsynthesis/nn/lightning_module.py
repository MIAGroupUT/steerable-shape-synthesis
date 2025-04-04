"""Base lightning module for implicit neural representations"""

from lightning import LightningModule
import torch
import numpy as np
from thyroidsynthesis.nn.models import LipschitzMLP
from tqdm import tqdm, trange
from skimage import measure
import trimesh
import point_cloud_utils as pcu


class INRLightningModule(LightningModule):
    def __init__(
        self,
        model,
        latent=None,
        lr=3e-4,
        clamp_distance=None,
        clamp_start_epoch=100,
        apply_correlation_loss=False,
        apply_batch_correlation_loss=False,
        correlation_loss_prefactor=1,
        apply_latent_loss=False,
        latent_loss_prefactor=1e-2,
        positional_embedding=None,
    ):
        """
        latent: ...
        lr: float, learning rate
        clamp_distance: float
            This clamps both the predicted and ground truth sdf, lower
            values make the model concentrate more on the surface and
            prevents from overfitting on sdf values far from the surface.
            Set to None to disable.
        """
        super().__init__()
        self.model = model
        self.loss = torch.nn.MSELoss()

        # Set up latent conditioning vectors
        if latent is None:
            self.latent_codes = None
        if type(latent) is np.ndarray:
            self.latent_codes = torch.nn.Embedding.from_pretrained(torch.tensor(latent))
        if type(latent) is tuple:
            self.latent_codes = torch.nn.Embedding(latent[0], latent[1])
            self.latent_codes.weight.data *= 0.01

        # If the latent variable is a list, it can be a composition of
        # - arrays, which means fixed preset values
        # - tuples, which means trainable values of that size
        if type(latent) is list:
            self._setup_latent_from_list(latent)
            self.latent_codes = self._get_latent_from_list

        self.save_hyperparameters(ignore=["model", "latent_codes"])

        if clamp_distance is not None:
            print(f"Training with clipping distance: {clamp_distance}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        shape_id, points, ground_truth_sdf = batch

        # Add latent codes
        if self.latent_codes is not None:
            latent_code = self.latent_codes(shape_id)[:, None, :].repeat(
                1, points.shape[1], 1
            )
            points = torch.cat((points, latent_code), dim=2).type(points.dtype)

        predicted_sdf, _ = self(points)
        if self.hparams.clamp_distance is not None:
            if self.current_epoch > self.hparams.clamp_start_epoch:
                clamp_dist = self.hparams.clamp_distance
            else:
                fraction = (
                    self.hparams.clamp_start_epoch - self.current_epoch
                ) / self.hparams.clamp_start_epoch
                clamp_dist = (1 - fraction) * self.hparams.clamp_distance + fraction * 1

            predicted_sdf = torch.clamp(predicted_sdf, -clamp_dist, clamp_dist)
            ground_truth_sdf = torch.clamp(
                ground_truth_sdf,
                -clamp_dist,
                clamp_dist,
            )

        loss = self.loss(predicted_sdf, ground_truth_sdf)

        self.log("train_loss", loss)

        # Add Lipschitz regularization if model is LipschitzMLP
        if type(self.model) is LipschitzMLP:
            lipschitz_loss = 1e-7 * self.model.get_lipschitz_loss()
            self.log("lipschitz_loss", lipschitz_loss)
            loss += lipschitz_loss

        # Optionally add latent regularization
        if self.hparams.apply_latent_loss:
            latent_code = self.latent_codes(shape_id)
            latent_loss = (
                self.hparams.latent_loss_prefactor
                * torch.norm(latent_code, dim=-1) ** 2
            )
            latent_loss = torch.mean(latent_loss)
            loss += latent_loss

        # Optionally add batched correlation loss
        if self.hparams.apply_batch_correlation_loss:
            batch_correlation_loss = self._get_batch_correlation_loss(shape_id)
            batch_correlation_loss = (
                self.hparams.correlation_loss_prefactor * batch_correlation_loss
            )
            loss += batch_correlation_loss

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def on_train_epoch_end(self):
        # Apply correlation loss to all embeddings
        correlation_loss = self._get_correlation_loss()
        self.log("correlation_loss", correlation_loss)
        if self.hparams.apply_correlation_loss:
            correlation_loss = (
                correlation_loss * self.hparams.correlation_loss_prefactor
            )
            opt = self.optimizers()
            opt.zero_grad()
            correlation_loss.backward()
            opt.step()

        # Log latent size to trainable bits
        latent_std = self._get_latent_std()
        self.log("latent_std", latent_std)

        # If no target meshes are provided, don't calculate chamfer
        if self.trainer.datamodule.target_meshes is None:
            return

        if self.current_epoch % 1000 != 0:
            return

        target_meshes = self.trainer.datamodule.target_meshes
        n_structures = len(target_meshes[0])
        structure_labels = list(target_meshes[0].keys())

        reconstructed_meshes = []
        for idx, target_mesh in tqdm(
            enumerate(target_meshes), desc="Reconstructing training meshes"
        ):
            latent_code = self.latent_codes(torch.tensor(idx, device=self.device))
            try:
                mesh = self._generate_latent(
                    latent_code, n_structures, structure_labels, resolution=64
                )
                reconstructed_meshes.append(mesh)
            except Exception:
                # print("meshing failed")
                # print(e)
                reconstructed_meshes.append(None)

        # If meshing failed for a certain case, do not log mean chamfer
        if None in reconstructed_meshes:
            return

        mean_chamfer = self._get_mean_chamfer(target_meshes, reconstructed_meshes)
        for anatomy, value in mean_chamfer.items():
            self.log(f"train_chamfer_{anatomy}", value)

    def _get_mean_chamfer(self, target_meshes, reconstructed_meshes):
        chamfer_distance = {}
        for anatomy in target_meshes[0].keys():
            chamfer_distance[anatomy] = []

        for target_mesh, reconstructed_mesh in zip(target_meshes, reconstructed_meshes):
            for anatomy in target_mesh.keys():
                chamfer_distance[anatomy].append(
                    pcu.chamfer_distance(
                        target_mesh[anatomy].vertices,
                        reconstructed_mesh[anatomy].vertices,
                    )
                )
        for anatomy, distances in chamfer_distance.items():
            chamfer_distance[anatomy] = np.mean(distances)

        return chamfer_distance

    def _generate_latent(
        self, latent, n_structures, structure_labels, resolution=64, extent=1.1
    ):
        """Generate a mesh given a latent code"""
        sampled_sdf = self._sample_sdf(
            n_structures, resolution=resolution, extent=extent, latent=latent
        )
        mesh_dict = {}
        for idx, label in enumerate(structure_labels):
            verts, faces, _, _ = measure.marching_cubes(sampled_sdf[idx], 0)
            verts = (2 * extent * verts / (resolution - 1)) - extent
            mesh_dict[label] = trimesh.Trimesh(verts, faces)
        return mesh_dict

    def _sample_sdf(
        self,
        n_structures,
        resolution: int,
        extent: float,
        latent=None,
        verbose: bool = False,
    ):
        self.eval()

        out_vol = np.zeros((n_structures, resolution, resolution, resolution))
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
            coords = coords.to(device=self.device)

            # Add latent code
            if latent is not None:
                latent_code = torch.tensor(latent, device=self.device).repeat(
                    coords.shape[0], 1
                )
                coords = torch.cat((coords, latent_code), dim=1).type(coords.dtype)

            with torch.no_grad():
                out, _ = self(coords)

            final_output = np.reshape(
                out.cpu().numpy(), im_slice.shape[:2] + (n_structures,)
            ).T
            out_vol[:, :, :, z_it] = final_output

        return out_vol

    def _get_latent_from_list(self, idx):
        latent_code = [bit(idx) for bit in self.latent_bits]
        if self.hparams.positional_embedding is not None:
            for bit_idx in range(len(latent_code)):
                latent_bit = latent_code[bit_idx]
                embedding_size = self.hparams.positional_embedding[bit_idx]
                latent_bit = self._apply_positional_encoding(latent_bit, embedding_size)
                latent_code[bit_idx] = latent_bit

        latent_code = torch.cat(latent_code, dim=-1)

        return latent_code

    def _setup_latent_from_list(self, latent):
        self.latent_bits = []
        for bit in latent:
            if type(bit) is np.ndarray:
                latent_bit = torch.nn.Embedding.from_pretrained(torch.tensor(bit))
            if type(bit) is tuple:
                latent_bit = torch.nn.Embedding(bit[0], bit[1])
                latent_bit.weight.data *= 0.01
            self.latent_bits.append(latent_bit)
        self.latent_bits = torch.nn.Sequential(*self.latent_bits)

    def _apply_positional_encoding(self, latent, size):
        if size == 0:
            return latent

        freq_bands = 2.0 ** torch.linspace(0, size // 2 - 1, size // 2)
        encoded = []
        for freq in freq_bands:
            encoded.append(torch.sin(freq * np.pi * latent))
            encoded.append(torch.cos(freq * np.pi * latent))
        encoded = torch.cat(encoded, dim=-1)
        return encoded

    def _get_latent_loss(self):
        """
        Calculate L2 norm loss for all trainable bits
        """
        trainable_bits = []
        for idx, bit in enumerate(self.latent_bits):
            if bit.weight.requires_grad:
                trainable_bits.append(bit)

        loss = 0
        for trainable_bit in trainable_bits:
            trainable_latent = trainable_bit.weight
            loss += self.latent_loss(trainable_latent)

        return loss

    def _get_latent_std(self):
        """
        Calculate global latent std for trainable part

        I use this to log whether latent loss does something yes/no.
        """
        trainable_bits = []
        for idx, bit in enumerate(self.latent_bits):
            if bit.weight.requires_grad:
                trainable_bits.append(bit)

        return torch.std(trainable_bits[0].weight)

    def _get_correlation_loss(self):
        """
        Loop over all fixed bits and calculate correlation loss w.r.t.
        the trainable bits
        """
        fixed_bits = []
        trainable_bits = []
        for idx, bit in enumerate(self.latent_bits):
            if bit.weight.requires_grad:
                trainable_bits.append(bit)
            else:
                fixed_bits.append(bit)

        loss = 0
        for fixed_bit in fixed_bits:
            fixed_latent = fixed_bit.weight
            for trainable_bit in trainable_bits:
                trainable_latent = trainable_bit.weight
                loss += self.correlation_loss(trainable_latent, fixed_latent)

        return loss

    def _get_batch_correlation_loss(self, shape_id):
        """
        Loop over all fixed bits and calculate correlation loss w.r.t.
        the trainable bits
        """
        fixed_bits = []
        trainable_bits = []
        for idx, bit in enumerate(self.latent_bits):
            if bit.weight.requires_grad:
                trainable_bits.append(bit)
            else:
                fixed_bits.append(bit)

        loss = 0
        for fixed_bit in fixed_bits:
            fixed_latent = fixed_bit(shape_id)
            for trainable_bit in trainable_bits:
                trainable_latent = trainable_bit(shape_id)
                loss += self.correlation_loss(trainable_latent, fixed_latent)

        return loss

    def correlation_loss(self, z_trainable, fixed_feature):
        """
        Penalizes correlation between trainable latent variables and a fixed feature.
        Args:
            z_trainable: (batch_size, z_dim) - Trainable latent codes
            fixed_feature: (batch_size, 1) - Fixed feature (e.g., binary flag)
        Returns:
            Scalar loss value
        """
        z_mean = z_trainable.mean(dim=0, keepdim=True)  # Mean over batch
        f_mean = fixed_feature.mean(dim=0, keepdim=True)

        # Compute covariance
        cov = ((z_trainable - z_mean) * (fixed_feature - f_mean)).mean(dim=0)

        # Compute standard deviations
        std_z = (
            z_trainable.std(dim=0, unbiased=False) + 1e-6
        )  # Add small value to avoid div by zero
        std_f = fixed_feature.std(dim=0, unbiased=False) + 1e-6

        # Compute absolute Pearson correlation
        corr = torch.abs(cov / (std_z * std_f))

        return corr.mean()

    def latent_loss(self, z_trainable):
        """
        Penalizes large latent vectors

        Args:
            z_trainable: (batch_size, z_dim) - Trainable latent codes
            fixed_feature: (batch_size, 1) - Fixed feature (e.g., binary flag)
        Returns:
            Scalar loss value
        """
        loss = torch.mean(torch.norm(z_trainable, dim=0))

        return loss
