"""
Scipt for training an INR on thyroid data
"""

from pathlib import Path
from thyroidsynthesis.nn.inr import INR
from thyroidsynthesis.nn.models import MLP
import numpy as np
import pandas as pd
import argparse
import trimesh


class Autodecoder(INR):
    def __init__(self, depth=3, size=256, latent_size=256, *args, **kwargs):
        self.model = MLP(
            layers=[3 + latent_size] + [size] * depth + [kwargs["n_structures"]]
        )
        super().__init__(*args, **kwargs)


def main(data_dir, ckpt_path, fixed_features):
    epochs = 10_000
    trainable_latent_size = 64
    log = False
    apply_correlation_loss = False
    apply_batch_correlation_loss = True
    apply_latent_loss = True
    correlation_loss_prefactor = 1e-5
    latent_loss_prefactor = 1e-4
    learning_rate = 3e-4
    batch_size = 1

    sdf_dir = data_dir / "sdf"
    ply_dir = data_dir / "ply"

    included_anatomy = [
        "thyroid",
    ]

    # Load features
    df = pd.read_csv(data_dir / "features.csv")

    # Load cases
    data_paths = [p for p in sdf_dir.glob("*.npz")]
    data_paths.sort()
    target_meshes = [trimesh.load(ply_dir / f"{p.stem}.ply") for p in data_paths]

    # Preset latent codes
    latent = []
    latent_size = 0

    # Add fixed features
    for feature_label in fixed_features:
        if feature_label == "":
            continue
        if feature_label == "volume":
            volume = np.array(df.volume)
            latent.append(volume.reshape((len(volume), 1)))
            latent_size += 1
        if feature_label == "isthmus_area":
            isthmus_area = np.array(df.isthmus_area)
            latent.append(isthmus_area.reshape((len(isthmus_area), 1)))
            latent_size += 1
        if feature_label == "symmetry":
            symmetry = np.array(df.symmetry)
            latent.append(symmetry.reshape((len(symmetry), 1)))
            latent_size += 1

    # Add trainable features
    latent.append((len(data_paths), trainable_latent_size))
    latent_size += 64

    SSM = Autodecoder(
        n_structures=len(included_anatomy),
        structure_labels=included_anatomy,
        latent_size=latent_size,
    )
    SSM.fit(
        data_paths,
        target_meshes=target_meshes,
        latent=latent,
        wandb=log,
        epochs=epochs,
        lr=learning_rate,
        enable_checkpointing=log,
        apply_correlation_loss=apply_correlation_loss,
        apply_batch_correlation_loss=apply_batch_correlation_loss,
        correlation_loss_prefactor=correlation_loss_prefactor,
        apply_latent_loss=apply_latent_loss,
        latent_loss_prefactor=latent_loss_prefactor,
        batch_size=batch_size,
    )

    SSM.save(ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("ckpt_path")
    parser.add_argument(
        "--fixed_features",
        type=str,
        default="volume,isthmus_area,symmetry",
    )
    args = parser.parse_args()
    fixed_features = [item for item in args.fixed_features.split(",")]
    main(Path(args.data_dir), Path(args.ckpt_path), fixed_features)
