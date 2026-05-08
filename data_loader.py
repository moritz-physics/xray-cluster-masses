import pandas as pd
import numpy as np

class Loader:
    
    def __init__(self, image_path, catalog_path):
        self.image_path = image_path
        self.catalog_path = catalog_path

    def get_data(self):
        '''
        Given the image and catalogue path, this function returns the
        images and labels needed to start directly with model training.
        Returns:
        --------
        images: [index, energy-band-image]
        labels_z: [log10(mass), redshift, index]
        '''
        is_efedssim = True
        is_efedsobs = False

        images = pd.read_pickle(self.image_path)

        key_mass = 'm500_wl_final' if is_efedsobs else 'M500c' if is_efedssim else 'HALO_M500c'
        key_redshift = 'z_final' if is_efedsobs else 'z' if is_efedssim else 'redshift_R'

        catalog_df = pd.read_feather(self.catalog_path)
        
        # mass = np.log10(catalog_df[key_mass])
        # redshifts = catalog_df[key_redshift].values
        # indices = catalog_df.index.values

        # labels_z = np.transpose([mass, redshifts, indices])

        label_dict = {
            "mass": np.log10(catalog_df[key_mass].values),
            "redshift": catalog_df[key_redshift].values,
            "index": catalog_df.index.values,
        }

        for col in catalog_df.columns:
            if col in label_dict:
                continue
            label_dict[col] = catalog_df[col].values        
        return images["gsm_3dImgs"] , label_dict
