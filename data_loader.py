import pandas as pd
import numpy as np

class Loader:
    """Reads the cluster image cube and matching catalogue from disk."""

    def __init__(self, image_path, catalog_path):
        """Store paths to the .pickle image cube and the catalogue (feather format)."""
        self.image_path = image_path
        self.catalog_path = catalog_path

    def get_data(self):
        '''
        Load images and per-cluster features so training can start directly.

        Returns
        -------
        images : list/array
            The 3D image cube under the pickle's "gsm_3dImgs" key.
            Shape (N, 50, 50, 10) — N clusters, 50x50 pixels, 10 energy bands.
        label_dict : dict[str, np.ndarray]
            Per-cluster feature dict containing:
              - "mass":     log10(M500c / M_sun)
              - "redshift": z
              - "index":    catalogue row index
              - plus every remaining column of the catalogue, verbatim.
        '''
        # The data variant flags below select which catalogue column names to use
        # for mass and redshift. This codebase only ever runs against the eFEDS
        # simulated catalogue, so the other branches are dead but kept for parity
        # with the upstream loader.
        is_efedssim = True   # eFEDS mock simulation (this project)
        is_efedsobs = False  # real eFEDS observations (not used here)

        images = pd.read_pickle(self.image_path)

        key_mass = 'm500_wl_final' if is_efedsobs else 'M500c' if is_efedssim else 'HALO_M500c'
        key_redshift = 'z_final' if is_efedsobs else 'z' if is_efedssim else 'redshift_R'

        catalog_df = pd.read_feather(self.catalog_path)

        label_dict = {
            "mass": np.log10(catalog_df[key_mass].values),
            "redshift": catalog_df[key_redshift].values,
            "index": catalog_df.index.values,
        }

        for col in catalog_df.columns:
            if col in label_dict:
                continue
            label_dict[col] = catalog_df[col].values
        return images["gsm_3dImgs"], label_dict
