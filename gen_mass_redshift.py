"""Catalogue exploration script: probes the pickle/feather data files and
generates per-feature histograms in `all_plots/`."""

import pickle
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import os


# Probe the image pickle: what's in it, what types and lengths?
with open("physics_data/eFEDS_01to18-3dImgs-7946clus-300pix-50pix-ext_det_thr.pickle", "rb") as f:
    data = pickle.load(f)

print(type(data))  # expect: dict
print(data.keys())

for key in data.keys():
    print(f"data[{key!r}] is a {type(data[key]).__name__} of length {len(data[key])}")

print(f"type of data['cluster'][0]:    {type(data['cluster'][0])}")
print(f"type of data['gsm_3dImgs'][0]: {type(data['gsm_3dImgs'][0])}")
print(f"len  of data['gsm_3dImgs'][0]: {len(data['gsm_3dImgs'][0])}")

print()
print("Now inspecting the catalogue file...")
print()

df = pd.read_feather("physics_data/eFEDS_mock_clusters_catalog_01to18-ext_det_thr.f")
print(df.columns)
print(df.head())



# Re-load the catalogue from the prep-images directory used while generating plots.
df = pd.read_feather("physics_prep_images/eFEDS_mock_clusters_catalog_01to18-ext_det_thr.f")

# Create output folder
output_dir = "all_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


column_labels = {
    'SRC_ID_IN': "Input Source ID",
    'RA': "Right Ascension (RA)",
    'DEC': "Declination (Dec)",
    'z': "Redshift (z)",
    'kT': "X-ray Temperature (keV)",
    'dL': "Luminosity Distance (cm)",
    'Lx_soft': "X-ray Luminosity (soft band)",
    'Lx_soft_2R500': "X-ray Luminosity (soft, 2×R500)",
    'Fx_soft': "X-ray Flux (soft band)",
    'Fx_soft_2R500': "X-ray Flux (soft, 2×R500)",
    'M500c': "Cluster Mass $M_{500c}$ [$M_\\odot$]",
    'R500c_arcmin': "R500 (arcmin)",
    'R500c_kpc': "R500 (kpc)",
    'pid': "Halo ID (pid)",
    'Mvir': "Virial Mass $M_{vir}$ [$M_\\odot$]",
    'Rvir': "Virial Radius $R_{vir}$ [kpc]",
    'Xoff': "Cluster Offset $X_{off}$",
    'b_to_a_500c': "Ellipticity $b/a$ at R500",
    'ID_SRC_OUT': "Detected Source ID",
    'RA_eSASS': "RA (eSASS Detection)",
    'DEC_eSASS': "Dec (eSASS Detection)",
    'RADEC_ERR': "Position Error (arcsec)",
    'EXT': "Extent (arcsec)",
    'EXT_ERR': "Extent Error",
    'EXT_LIKE': "Extent Likelihood",
    'ML_RATE': "Maximum Likelihood Rate",
    'ML_RATE_ERR': "Rate Error",
    'DET_LIKE': "Detection Likelihood",
    'ML_BKG': "Background Count Rate",
    'ML_EXP': "Exposure (s)",
    'input_SRC_ID': "Original Source ID",
    'Nreal': "Number of Realizations",
    'CLU_X': "Image X Position",
    'CLU_Y': "Image Y Position",
}


for col, label in column_labels.items():
    plt.figure()
    plt.hist(df[col], bins=30)
    plt.xlabel(label)
    plt.ylabel("Number of Clusters")
    plt.title(f"Distribution of {label}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{col}_hist.png"))
    print(f"fig : {col}_hist.png is saved")
    plt.close()



