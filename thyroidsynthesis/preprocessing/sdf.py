"""Script which samples random points on mesh surfaces and calculates
the sdf."""

import trimesh
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
from pysdf import SDF


def sample_points_on_surface_noise(n_points, mesh, std=0.1):
    n_points_on_mesh = n_points // 5 * 4
    n_points_off_mesh = n_points - n_points_on_mesh

    # Sample points on mesh surfaces
    points_on_surface, _ = trimesh.sample.sample_surface(mesh, n_points_on_mesh)

    # Sample points normally distributed around surface points
    indices = np.random.choice(n_points_on_mesh, n_points_off_mesh)
    points_off_surface = points_on_surface[indices] + np.random.normal(
        scale=std, size=(n_points_off_mesh, 3)
    )

    # Combine all points
    points = np.concatenate([points_on_surface, points_off_surface])

    return points


def calculate_sdf_for_mesh(mesh, points):
    """Calculate SDF for all points on a single mesh"""
    f = SDF(mesh.vertices, mesh.faces)
    return f(points)


def main(
    ply_dir,
    output_dir,
    n_points=50000,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all meshes
    mesh_paths = list(ply_dir.glob("*.ply"))
    meshes = [trimesh.load(p) for p in mesh_paths]

    # SDF loop
    for mesh_path, mesh in tqdm(
        zip(mesh_paths, meshes), desc="Sampling SDF values", total=len(meshes)
    ):
        file_name = mesh_path.name
        dest_path = (output_dir / file_name).with_suffix(".npz")

        # Sample points
        points = sample_points_on_surface_noise(n_points, mesh)

        # Calculate SDF
        sdf_column = calculate_sdf_for_mesh(mesh, points)

        # Stack results into column
        sdf = np.array([sdf_column]).T

        # Save results
        np.savez(dest_path, points=points, sdf=sdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    main(Path(args.ply_dir), Path(args.output_dir))
