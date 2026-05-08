# X-ray Galaxy Cluster Mass Inference with CNNs

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Convolutional neural networks for estimating galaxy cluster masses (with calibrated uncertainties) directly from simulated eROSITA X-ray photon-count images.

> **Group project.** Originally developed as a team lab project at Ludwig-Maximilians-Universität München (LMU), Faculty of Physics, AI in Physics Research Group. The full team report (LaTeX source + PDF) is in [`AI Lab 4 - xRay/`](AI%20Lab%204%20-%20xRay/main.pdf). This repository has been migrated for portfolio purposes without the original commit history.

## Overview

This project trains CNN regression models that map multi-channel X-ray photon-count images of galaxy clusters to a posterior over the cluster mass. The training data comes from a mock eROSITA-eFEDS catalogue: ~7,946 simulated galaxy clusters, each represented as a 50×50 pixel image with 10 energy bands. The networks predict both a mean mass and a per-cluster variance via a Gaussian negative-log-likelihood loss, so each prediction comes with a calibrated 1σ uncertainty. We also explore auxiliary scalar inputs (redshift, gas temperature `kT`), data augmentation, and Optuna-based hyperparameter search.

## Physics Background

Galaxy clusters are the most massive gravitationally-bound structures in the universe — agglomerations of hundreds to thousands of galaxies embedded in a halo of dark matter and a hot (10⁷–10⁸ K) intra-cluster medium (ICM). Because the ICM is so hot, it radiates strongly in X-rays via thermal bremsstrahlung, which makes X-ray imaging a powerful and largely unbiased probe of cluster structure. The total mass of a cluster is the single most important observable for cosmology — its abundance as a function of mass and redshift constrains σ₈, Ωₘ, and dark-energy parameters — but mass cannot be measured directly. Traditional approaches infer mass from scaling relations between mass and X-ray observables (luminosity, temperature, gas mass), which require strong assumptions (hydrostatic equilibrium, spherical symmetry) and break down for disturbed clusters. A CNN, by contrast, can learn the mass mapping straight from the 2D photon-count image, implicitly using morphology, profile shape, and surface-brightness information that scaling relations discard.

## ML Pipeline

### Data
- **Inputs:** 3D image arrays of shape `(N, 50, 50, 10)` — N=7,946 clusters, 50×50 pixel cutouts in 10 X-ray energy bands (stored in a `.pickle` file).
- **Labels:** `log10(M500c / M_sun)` — the cluster mass within a sphere whose mean overdensity is 500× the critical density, taken from a feather/FITS catalogue. Redshift `z` and ICM temperature `kT` are also available as auxiliary inputs.
- **Splits:** 70% train / 15% val / 15% test, fixed seed (`random_state=42`). See `preprocessing.Preprocessor.split`.
- **Normalisation:** Images are normalised per-image (zero mean, unit variance over all pixels and channels). Scalar features (mass, redshift, kT) are standardised against the training distribution; predictions are denormalised before plotting.

### CNN architecture
The architecture is fully configurable via dictionaries (`models.ModelFactory.build_cnn`):
- A stack of `Conv2D` layers (filter count, kernel size, optional batchnorm and max-pooling per layer).
- A `Flatten` followed by dense layers with optional dropout/batchnorm.
- An optional **auxiliary branch**: scalar inputs (redshift, kT) are concatenated with the flattened CNN features and passed through a small "head" of dense layers.
- A final `Dense(2)` outputs raw `(μ, raw_σ²)`. A `Lambda` layer applies `sigmoid` to the variance channel to keep it positive and bounded, giving the final `(μ, σ²)` prediction.

The best architecture (from Optuna, see below) has 3 conv layers (32→32→128 filters), 2 dense layers (64→256, dropout ~0.4), a 256-unit head, and ~20.6M parameters.

### Custom loss: Gaussian negative log-likelihood
Instead of MSE, the network is trained with a heteroscedastic Gaussian NLL loss, so it learns to predict an input-dependent variance:

$$
\mathcal{L}(y, \mu, \sigma^2) = \frac{1}{2}\log(\sigma^2 + \epsilon) + \frac{(y - \mu)^2}{2(\sigma^2 + \epsilon)}
$$

Predicting a small σ² when wrong is heavily penalised by the second term; predicting a huge σ² to game the second term is penalised by the first (log) term. The minimum is achieved when σ² matches the actual squared error. The implementation is in `training.Trainer.loss_nll`. We also track the MSE on the mean as an interpretable metric (`Trainer.mse_mean`).

### Uncertainty calibration
Because the network outputs a variance per cluster, we get a built-in 1σ confidence interval per prediction without needing a full Bayesian network or MC dropout. We measure calibration via **coverage**: the fraction of test examples where the true mass lies inside `[μ − σ, μ + σ]`. For a well-calibrated Gaussian this should be ≈68%. See `Trainer.check_uncertainty` and the "Pull" residual `(μ − y_true)/σ` in the prediction plots.

### Data augmentation
For training-set augmentation we apply random horizontal/vertical flips, ±10% rotations, and ±10% zoom (`preprocessing.Preprocessor.augment_images`, using Keras preprocessing layers). Random crop+resize was also tested but slightly degraded results and is kept disabled. Augmentation is applied once before training (not on-the-fly).

### Hyperparameter search with Optuna
`optuna_tuner.py` runs a TPE-sampler search over: number of conv/dense/head layers, filter counts, kernel sizes, batchnorm, max-pooling, dropout rates, learning rate, choice of auxiliary feature subset (`[]`, `[redshift]`, `[kT]`, `[Lx_soft]`, …), and whether to augment. Each trial trains for up to 100 epochs with early stopping (patience 20). Results are persisted to a SQLite study DB and a CSV checkpoint every 10 trials. The best trial used redshift + kT as auxiliary inputs, augmentation on, learning rate ≈ 2.6e-4 (full hyperparameters in `results_optuna/best_trial.txt`).

## Results

We trained eight model variants on `final_results/` (NLL loss, with calibrated uncertainty) and five earlier variants on `model_results/` (MSE loss only, no uncertainty), each combining different choices of auxiliary features (none, redshift, redshift+kT) and augmentation (on/off). Each variant produces a predicted-vs-true mass plot with ±1σ error bars and a residual "pull" panel, plus the corresponding training curves.

### Best model — Optuna-tuned, augmentation, redshift + kT (`final_results/aug_red_kT_optuna/`)

![Best model — predictions](final_results/aug_red_kT_optuna/mass_pred_test.png)

*Predicted vs. true `log10(M500c/M_sun)` on the held-out test set. Points are coloured blue if the truth lies within the predicted ±1σ interval, orange otherwise. The shaded band shows the ±1σ region around y=x. The lower panel is the pull `(μ − y_true)/σ`; a well-calibrated model has pulls scattered around 0 with std ≈ 1.*

![Best model — training curves](final_results/aug_red_kT_optuna/metrics_vs_epoch.png)

*Train/val MSE (left) and NLL (right) against epoch. Red marker = best validation epoch (early-stopping checkpoint).*

### Final-results variants (NLL loss, calibrated uncertainty)

#### `simplest` — image-only, MSE loss baseline reproduced under NLL

![simplest — predictions](final_results/simplest/mass_pred_test.png) ![simplest — training curves](final_results/simplest/metrics_vs_epoch.png)

#### `simple` — image-only, default architecture

![simple — predictions](final_results/simple/mass_pred_test.png) ![simple — training curves](final_results/simple/metrics_vs_epoch.png)

#### `aug_simple` — image-only + augmentation

![aug_simple — predictions](final_results/aug_simple/mass_pred_test.png) ![aug_simple — training curves](final_results/aug_simple/metrics_vs_epoch.png)

#### `redshift` — image + redshift auxiliary input

![redshift — predictions](final_results/redshift/mass_pred_test.png) ![redshift — training curves](final_results/redshift/metrics_vs_epoch.png)

#### `aug_redshift` — image + redshift + augmentation

![aug_redshift — predictions](final_results/aug_redshift/mass_pred_test.png) ![aug_redshift — training curves](final_results/aug_redshift/metrics_vs_epoch.png)

#### `redshift_kT` — image + redshift + kT

![redshift_kT — predictions](final_results/redshift_kT/mass_pred_test.png) ![redshift_kT — training curves](final_results/redshift_kT/metrics_vs_epoch.png)

#### `aug_redshift_kT` — image + redshift + kT + augmentation

![aug_redshift_kT — predictions](final_results/aug_redshift_kT/mass_pred_test.png) ![aug_redshift_kT — training curves](final_results/aug_redshift_kT/metrics_vs_epoch.png)

### Earlier variants (MSE loss, no predicted uncertainty)

These models were trained before the switch to a Gaussian NLL loss; they predict a point estimate only.

#### `model_results/simple`
![model_results/simple — predictions](model_results/simple/mass_pred_test.png) ![model_results/simple — MSE](model_results/simple/mse_vs_epoch.png)

#### `model_results/redshift`
![model_results/redshift — predictions](model_results/redshift/mass_pred_test.png) ![model_results/redshift — MSE](model_results/redshift/mse_vs_epoch.png)

#### `model_results/aug_redshift`
![model_results/aug_redshift — predictions](model_results/aug_redshift/mass_pred_test.png) ![model_results/aug_redshift — MSE](model_results/aug_redshift/mse_vs_epoch.png)

#### `model_results/redshif_kT` *(sic — folder name preserved)*
![model_results/redshif_kT — predictions](model_results/redshif_kT/mass_pred_test.png) ![model_results/redshif_kT — MSE](model_results/redshif_kT/mse_vs_epoch.png)

#### `model_results/aug_redshift_kT`
![model_results/aug_redshift_kT — predictions](model_results/aug_redshift_kT/mass_pred_test.png) ![model_results/aug_redshift_kT — MSE](model_results/aug_redshift_kT/mse_vs_epoch.png)

### Catalogue distributions
Marginal distributions of every catalogue column (mass, redshift, kT, fluxes, positions, …) are auto-generated by `gen_mass_redshift.py` and saved to `all_plots/`. A few key ones:

![Mass distribution](all_plots/M500c_hist.png) ![Redshift distribution](all_plots/z_hist.png) ![Temperature distribution](all_plots/kT_hist.png)

### Example image
What a single cluster looks like in the 10 eROSITA energy bands:

![Example multi-band cluster image](images/efeds_energy_image.png)

## Repository Structure

```
xray-cluster-masses/
├── README.md
├── pyproject.toml                  # uv-managed dependencies (Python ≥ 3.12)
├── uv.lock
│
├── data_loader.py                  # Loader: reads pickle of images + feather/FITS catalogue
├── preprocessing.py                # Preprocessor: normalise / split / augment
├── models.py                       # ModelFactory.build_cnn — configurable CNN with optional aux inputs
├── training.py                     # Trainer: NLL loss, training loop, plotting, coverage
├── main.py                         # CLI entry point: end-to-end train + evaluate
├── optuna_tuner.py                 # Optuna TPE hyperparameter search
├── gen_mass_redshift.py            # Catalogue exploration → all_plots/
├── available_features.txt          # Reference list of catalogue columns
│
├── notebooks/
│   └── walkthrough.ipynb           # End-to-end walkthrough notebook
│
├── final_results/                  # Models trained with Gaussian NLL loss
│   ├── simplest/   simple/   aug_simple/
│   ├── redshift/   aug_redshift/
│   ├── redshift_kT/   aug_redshift_kT/
│   └── aug_red_kT_optuna/          # Best model (Optuna-tuned)
│
├── model_results/                  # Earlier models (MSE loss, no uncertainty)
│   ├── simple/   redshift/   aug_redshift/
│   ├── redshif_kT/   aug_redshift_kT/
│
├── results_optuna/                 # Optuna study DB, best_trial.txt, model summary
├── optuna_results/                 # Earlier Optuna run (CSV + summary)
│
├── all_plots/                      # Per-feature catalogue histograms
├── plots_of_data_loader/           # Mass/redshift histograms after loading
├── images/                         # Reference images (example energy-band image, etc.)
│
└── AI Lab 4 - xRay/                # LaTeX source + PDF of the team report
```

## Data

Two large data files (~2 GB total) are required and **are not committed to the repo**.

| File | Description | Size |
|---|---|---|
| `eFEDS_01to18-3dImgs-7946clus-300pix-50pix-ext_det_thr.pickle` | Pickle dict with key `gsm_3dImgs` → list of `(50, 50, 10)` photon-count cutouts and `cluster` IDs (7,946 simulated clusters). | ~1.8 GB |
| `eFEDS_mock_clusters_catalog_01to18-ext_det_thr.fits` | Catalogue of cluster properties: mass `M500c`, redshift `z`, temperature `kT`, soft-band flux/luminosity, position, ellipticity, eSASS detection metadata, etc. (see `available_features.txt`). | ~few MB |

### Download

```bash
mkdir -p physics_data
cd physics_data

wget https://cloud.physik.lmu.de/index.php/s/ParyrwygTYdFcWY/download/eFEDS_01to18-3dImgs-7946clus-300pix-50pix-ext_det_thr.pickle
wget https://cloud.physik.lmu.de/index.php/s/5xex5aXqRYSdSMw/download/eFEDS_mock_clusters_catalog_01to18-ext_det_thr.fits
```

### Catalogue format note

The `.py` pipeline (`data_loader.py`) expects the catalogue in **feather** format with extension `.f`. Convert the downloaded FITS once:

```python
from astropy.table import Table
import pandas as pd

df = Table.read(
    "physics_data/eFEDS_mock_clusters_catalog_01to18-ext_det_thr.fits"
).to_pandas()
df.to_feather(
    "physics_data/eFEDS_mock_clusters_catalog_01to18-ext_det_thr.f"
)
```

The walkthrough notebook (`notebooks/walkthrough.ipynb`) reads the FITS file directly via `astropy` and does not require this conversion.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. Python ≥ 3.12 is required.

```bash
# clone
git clone <this-repo>
cd xray-cluster-masses

# install dependencies into a local .venv
uv sync

# launch JupyterLab and open notebooks/walkthrough.ipynb
uv run jupyter lab
```

To run the CLI pipeline (after downloading + converting the data):

```bash
# image-only model
uv run python main.py --out-dir simple

# image + redshift + augmentation
uv run python main.py --features redshift --augment-images --out-dir aug_redshift

# Optuna hyperparameter search
uv run python optuna_tuner.py --n-trials 50 --out-dir results_optuna
```

## References

- C. L. Sarazin, *X-ray Emission from Clusters of Galaxies* — review article: <https://ned.ipac.caltech.edu/level5/March02/Sarazin/frames.html>
- Liu et al. 2022, *The eROSITA Final Equatorial-Depth Survey (eFEDS) — A multi-wavelength view of the X-ray-selected galaxy cluster sample*: <https://arxiv.org/abs/2008.08404>
- Comparat et al. 2020, *Full-sky X-ray surveys with eROSITA — synthetic catalogues used here*: <https://arxiv.org/abs/2106.14528>
- I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.

## License

MIT — see [LICENSE](LICENSE) if present, otherwise the badge is the canonical statement of intent.
