import pickle
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import os 


with open("physics_data/eFEDS_01to18-3dImgs-7946clus-300pix-50pix-ext_det_thr.pickle", "rb") as f:
    data = pickle.load(f)

# What is this thing?
print(type(data)) # it is a dic

print(data.keys())

for key in data.keys():
    # what type are the kyes?
    print(f"The {key} is of {type(data[key])} type ") # they are lists
    print(f"The len of {key} is {len(data[key])}")


print(f" what is the type of the elemetns of the data['cluster'] ",type(data['cluster'][0]))
print(f" what is the type of the elemetns of the data['gsm_3dImgs'] ",type(data['gsm_3dImgs'][0]))
print(f" what is the length of the elemetns of the data['gsm_3dImgs'] ",len(data['gsm_3dImgs'][0]))

print()
print("lETS LOOK AT THE OTHER FILE!")
print()

df = pd.read_feather("physics_data/eFEDS_mock_clusters_catalog_01to18-ext_det_thr.f")
print(df.columns)
print(df.head())



# Load data
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



