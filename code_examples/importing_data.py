import pandas as pd
import numpy as np

#adapt this
# data_directory = '/home/a/Arvanitis.Vyron/xray-clusters-group-a/xray-clusters-group-a/physics_data'
# image_file = 'eFEDS_01to18-3dImgs-7946clus-300pix-50pix-ext_det_thr.pickle'
# catalog_file = 'eFEDS_mock_clusters_catalog_01to18-ext_det_thr.f'

def get_data(image_p, catalog_f):
    '''
    Given the image and catalogue path, this function returns the
    images and labels needed to start directly with model training.
    train_labels: Can be used to filter a particular mass range
    Returns:
    images: [index, energy-band-image]
    labels_z: [log10(mass), redshift, index]
    '''
    is_efedssim=True
    is_efedsobs=False
    images = pd.read_pickle(image_p)
    key_mass = 'm500_wl_final' if is_efedsobs else 'M500c' if is_efedssim else 'HALO_M500c'
    key_redshift = 'z_final' if is_efedsobs else 'z' if is_efedssim else 'redshift_R'
    catalog_df = pd.read_feather(catalog_f)
    labels = np.log10(catalog_df[key_mass])
    redshifts = catalog_df[key_redshift].values
    indices = catalog_df.index.values
    labels_z = np.transpose([labels, redshifts, indices])
    return images, labels_z

# images, labels = get_data(image_p=data_directory+image_file, catalog_f=data_directory+catalog_file)
