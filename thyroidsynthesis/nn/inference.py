"""Do inference with a model trained with the train.py script.

This does the following:
    - Reconstruct training meshes and evaluate chamfer distance
    - Generate 1000 random meshes and compute volume, isthmus area and symmetry
"""

from pathlib import Path
import numpy as np
import trimesh
import torch
from tqdm import tqdm
import point_cloud_utils as pcu
import pandas as pd
import argparse
import scipy
from thyroidsynthesis.nn.inr import INR
from thyroidsynthesis.nn.models import MLP
from thyroidsynthesis.preprocessing.features import (
    volume,
    isthmus_area,
    symmetry,
)

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


class Autodecoder(INR):
    def __init__(self, depth=3, size=256, latent_size=256, *args, **kwargs):
        self.model = MLP(
            layers=[3 + latent_size] + [size] * depth + [kwargs["n_structures"]]
        )
        super().__init__(*args, **kwargs)


def reconstruct(output_dir, SSM, target_meshes):
    dest_dir = output_dir / "reconstructed_meshes"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for idx, target_mesh in tqdm(
        enumerate(target_meshes),
        desc="Reconstructing training meshes",
        total=len(target_meshes),
    ):
        latent_code = SSM.lightning_module.latent_codes(
            torch.tensor(idx).to(device=DEVICE)
        )
        mesh = SSM.generate_latent(latent_code, resolution=64)["thyroid"]
        mesh.export(dest_dir / target_mesh.metadata["file_name"])


def chamfer(output_dir, target_meshes):
    chamfer_distance = {}
    for target_mesh in tqdm(target_meshes, desc="Calculating chamfer distance"):
        mesh_id = target_mesh.metadata["file_name"][:5]
        mesh_path = (
            output_dir / "reconstructed_meshes" / target_mesh.metadata["file_name"]
        )
        reconstructed_mesh = trimesh.load(mesh_path)
        chamfer_distance[mesh_id] = pcu.chamfer_distance(
            target_mesh.vertices, reconstructed_mesh.vertices
        )

    df = pd.DataFrame.from_dict(
        chamfer_distance, orient="index", columns=["chamfer_distance"]
    )
    df.to_csv(output_dir / "chamfer_distance.csv", index_label="id")


def generate(output_dir, SSM):
    mesh_dir = output_dir / "generated_meshes"
    code_dir = output_dir / "generated_meshes_codes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    n_random_samples = 1000
    sampler = get_random_latent_code_sampler(SSM)
    for idx in tqdm(range(n_random_samples), desc="Generating random meshes"):
        success = False
        while not success:
            latent_code_numpy = sampler()
            latent_code = torch.tensor(latent_code_numpy, device=DEVICE)
            try:
                mesh = SSM.generate_latent(latent_code, resolution=64)["thyroid"]
                mesh.export(mesh_dir / f"{idx:04d}.ply")
                success = True
            except Exception as e:
                print("meshing failed")
                print(e)
                pass
        np.save(code_dir / f"{idx:04d}.npy", latent_code_numpy)


def get_random_latent_code_sampler(SSM):
    full_weight_matrix = np.concatenate(
        [bit.weight.cpu().detach().numpy() for bit in SSM.lightning_module.latent_bits],
        axis=1,
    )

    n_features = full_weight_matrix.shape[1]

    if n_features > 64:
        n_fixed_features = n_features - 64
        fixed_features = [full_weight_matrix[:, i] for i in range(n_fixed_features)]
        full_weight_matrix = full_weight_matrix[:, n_fixed_features:]

    # Trainable features get normally distributed samples
    mean = np.mean(full_weight_matrix, axis=0)
    std = np.std(full_weight_matrix, axis=0)
    rng = np.random.default_rng()

    # Other features get histogram features
    if n_features > 64:
        hists = [
            scipy.stats.rv_histogram(np.histogram(f), density=False)
            for f in fixed_features
        ]

    def sampler():
        code = []
        if n_features > 64:
            code.append([hist.rvs() for hist in hists])
        code.append(rng.normal(mean, std))
        return np.concatenate(code)

    return sampler


def compute_features(output_dir):
    mesh_dir = output_dir / "generated_meshes"
    features = {}
    for mesh_path in tqdm(
        list(mesh_dir.glob("*.ply")), desc="Computing anatomical features"
    ):
        id = mesh_path.stem
        mesh = trimesh.load(mesh_path)
        features[id] = [volume(mesh), isthmus_area(mesh), symmetry(mesh)]

    df = pd.DataFrame.from_dict(
        features, orient="index", columns=["volume", "isthmus_area", "symmetry"]
    )
    df.to_csv(output_dir / "generated_features.csv", index_label="id")


def load_checkpoint(ckpt_path, latent_size):
    SSM = Autodecoder(
        n_structures=1,
        structure_labels=["thyroid"],
        latent_size=latent_size,
        file_path=ckpt_path,
    )
    return SSM


def load_target_meshes(data_dir):
    ply_dir = data_dir / "ply"

    # Load cases
    data_paths = [p for p in ply_dir.glob("*.ply")]
    data_paths.sort()
    target_meshes = [trimesh.load(p) for p in data_paths]

    return target_meshes


def collect_preset_features(output_dir, fixed_features):
    code_paths = list((output_dir / "generated_meshes_codes").glob("*.npy"))
    code_paths.sort()
    for feature_idx, feature_label in enumerate(fixed_features):
        feature = []
        for code_path in code_paths:
            code = np.load(code_path)
            feature.append(code[feature_idx])
        np.save(output_dir / f"{feature_label}.npy", feature)


def main(data_dir, ckpt_path, output_dir, fixed_features):
    output_dir.mkdir(parents=True, exist_ok=True)
    SSM = load_checkpoint(ckpt_path, 64 + len(fixed_features))
    target_meshes = load_target_meshes(data_dir)
    reconstruct(output_dir, SSM, target_meshes)
    chamfer(output_dir, target_meshes)
    generate(output_dir, SSM)
    compute_features(output_dir)
    collect_preset_features(output_dir, fixed_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("ckpt_path")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--fixed_features",
        type=str,
        default="volume,isthmus_area,symmetry",
    )
    args = parser.parse_args()
    fixed_features = [item for item in args.fixed_features.split(",")]
    main(
        Path(args.data_dir), Path(args.ckpt_path), Path(args.output_dir), fixed_features
    )
