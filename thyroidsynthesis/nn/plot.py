"""Make standard plots for each train_thyroid.py experiment."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.metrics import r2_score


def reconstruction_histogram(output_dir):
    df = pd.read_csv(output_dir / "chamfer_distance.csv")
    chamfer_distance = np.array(df.chamfer_distance)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.hist(chamfer_distance, bins=100)
    ax.set_title("Chamfer distance")
    return fig


def correlation(output_dir, feature_label, feature_idx):
    df = pd.read_csv(output_dir / "generated_features.csv")
    if feature_label == "volume":
        feature = np.array(df.volume)
        lims = [0, 0.5]
    if feature_label == "isthmus_area":
        feature = np.array(df.isthmus_area)
        lims = [0, 0.2]
    if feature_label == "symmetry":
        feature = np.array(df.symmetry)
        lims = [0, 1]

    preset_feature = np.load(output_dir / f"{feature_label}.npy")
    r2 = r2_score(preset_feature, feature)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.scatter(preset_feature, feature, alpha=0.5)
    ax.set_xlabel(f"Preset {feature_label}")
    ax.set_ylabel(f"Generated {feature_label}")
    ax.set_title(f"$R^2$ = {r2}")

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    return fig


def save_fig(fig, file_path):
    fig.savefig(file_path)


def main(output_dir, fixed_features):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    figs = {}
    figs["reconstruction_chamfer"] = reconstruction_histogram(output_dir)
    for idx, feature in enumerate(fixed_features):
        figs[f"correlation_{feature}"] = correlation(output_dir, feature, idx)

    for fig_name, fig in figs.items():
        save_fig(fig, plot_dir / f"{fig_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument(
        "--fixed_features",
        type=str,
        default="volume,isthmus_area,symmetry",
    )
    args = parser.parse_args()
    fixed_features = [item for item in args.fixed_features.split(",")]
    main(Path(args.output_dir), fixed_features)
