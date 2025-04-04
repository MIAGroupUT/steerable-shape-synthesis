"""Precompute the following features for the thyroid meshes:

- Volume
- Isthmus area
- Symmetry IoU
"""

import trimesh
from pathlib import Path
import pandas as pd
import argparse
from tqdm import tqdm


def symmetry(mesh):
    if not mesh.is_volume:
        return -1
    transformation_matrix = trimesh.transformations.reflection_matrix(
        [0, 0, 0], [0, 0, 1]
    )
    mesh2 = mesh.copy()
    mesh2.apply_transform(transformation_matrix)
    intersection = trimesh.boolean.intersection([mesh, mesh2])
    union = trimesh.boolean.union([mesh, mesh2])
    return intersection.volume / union.volume


def volume(mesh):
    return mesh.volume


def isthmus_area(mesh):
    slice = mesh.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
    if slice is None:
        return 0
    slice_2D, to_3D = slice.to_planar()
    return slice_2D.area


def main(ply_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all meshes
    mesh_paths = list(ply_dir.glob("*.ply"))
    meshes = [trimesh.load(p) for p in mesh_paths]

    features = {}
    for mesh_path, mesh in tqdm(
        zip(mesh_paths, meshes),
        desc="Calculating anatomical features",
        total=len(meshes),
    ):
        id = mesh_path.stem
        features[id] = [volume(mesh), isthmus_area(mesh), symmetry(mesh)]

    df = pd.DataFrame.from_dict(
        features, orient="index", columns=["volume", "isthmus_area", "symmetry"]
    )
    df.to_csv(output_dir / "features.csv", index_label="id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    main(Path(args.ply_dir), Path(args.output_dir))
