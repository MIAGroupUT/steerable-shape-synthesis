# Steerable Anatomical Shape Synthesis with Implicit Neural Representations

[![Paper](https://img.shields.io/badge/Paper-2504.03313-b31b1b.svg)](https://arxiv.org/abs/2504.03313) [![Data](https://img.shields.io/badge/Data-10.5281%20%2F%20zenodo.15100852-blue.svg)](https://doi.org/10.5281/zenodo.15100852)

This repository holds the code for the paper [*Steerable Anatomical Shape Synthesis with Implicit Neural Representations*](https://arxiv.org/abs/2504.03313).
In this work we train implicit neural representations of 3D thyroid glands, conditioned on a latent code consisting of a combination of fixed anatomical features and learned latent features.
The resulting model can be used generatively by sampling random latent codes, and can be steered through the conditioning features.

The figure below shows an example application, where we edit the volume, isthmus (bridge) area and symmetry of a single thyroid independently:

![](images/edit_plot.svg)

# Data

Access the data used in the paper via [zenodo](https://zenodo.org/records/15100852).

# Running the code

The code is layed out as a package.
The easiest way to work with it, is to setup a conda environment with the `environment.yml` provided in this repository:

```sh
conda env create -f environment.yml
```

And then install the repository as an editable package:

```sh
conda activate thyroidsynthesis
pip install --no-build-isolation --no-deps -e .
```

If you run Windows, we recommend using WSL.

## Pre-sampling SDF values

If you downloaded the Zenodo dataset, all presampled SDF values are already available as `.npz` files in the `sdf` directory.
If you want to change the sampling, or re-run it for other meshes, run the following:

```sh
python -m thyroidsynthesis.preprocessing.sdf /path/to/ply /path/to/sdf
```

## Pre-computing anatomical features

If you downloaded the Zenodo dataset, all presampled thyroid features values are already available in the `features.csv` file.
If you want to re-run this for other meshes, run the following:

```sh
python -m thyroidsynthesis.preprocessing.features /path/to/ply /path/to/output_dir
```

This will write a `features.csv` file to `/path/to/output_dir`.

## Training

To train a model, run:

```sh
python -m thyroidsynthesis.nn.train /path/to/data /path/to/checkpoint.ckpt --fixed_features volume,isthmus_area,symmetry
```

This will train a model using training settings as reported in the paper, and will save the final model to `/path/to/checkpoint.ckpt`.

The data folder should look like:

```
/path/to/data
	|
	ply/
		|
		001.ply
		002.ply
		...
	sdf/
		|
		001.npz
		002.npz
		...
```

If you download our data from zenodo and unzip, this is already the case.

## Inference

For inference, run:

```sh
python -m thyroidsynthesis.nn.inference /path/to/data /path/to/checkpoint.ckpt /path/to/results --fixed_features volume,isthmus_area,symmetry
```

This will reconstruct and generate thyroid meshes, and save them in `/path/to/results`.

## Plot

To plot some basic analysis, similar to what is shown in the paper, run:

```sh
python -m thyroidsynthesis.nn.plot /path/to/results
```

# Citation

If you use this code, please cite our arXiv paper:

```
@misc{dewilde2025steerableanatomicalshapesynthesis,
      title={Steerable Anatomical Shape Synthesis with Implicit Neural Representations}, 
      author={Bram de Wilde and Max T. Rietberg and Guillaume Lajoinie and Jelmer M. Wolterink},
      year={2025},
      eprint={2504.03313},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.03313}, 
}
```
