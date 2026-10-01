# instrument_config.py
from dataclasses import dataclass

import astropy.units as units
import astropy.constants as const
from astropy.io import fits
import numpy as np

"""
@dataclass(frozen=True)
class XIFU_Config:
    pixel_size_m: float = 317e-6  #meters
    athena_focal_length: float = 12.0 #meters
    pointing_shape: tuple[int, int] = (58,58)

    @property
    def pixsize_arcsec(self) -> float:
        Convert pixel size to arcsec.
        return (self.pixel_size_m / self.athena_focal_length * units.radian).to(units.arcsec)
"""

class XIFU_Config:

    """
    Class created to store the constants used throughout the simulation
    """
    def __init__(self):
        r"""
        - Pixel size = $317\cdot10^{-6}$m
        - Focal length = 12m
        - Pointing shape in pixels 58x58
        - Number of pixels = 1504

        """
        self.pixel_size_m = 317e-6  #meters
        self.athena_focal_length = 12.0 #meters
        self.pointing_shape = (58,58)

        self.pixsize_arcsec = (self.pixel_size_m / self.athena_focal_length * units.radian).to(units.arcsec)
        self.pixsize_degree = (self.pixel_size_m / self.athena_focal_length * units.radian).to(units.degree)

        self.xifu_pixel_number = 1504

        self.std_xmlfile = '/xifu/usr/share/sixte/instruments/new-athena-xifu/baseline/xifu_nofilt_infoc.xml'
        self.vign_file_path = '/xifu/usr/share/sixte/instruments/new-athena-xifu/instdata/new_athena_xifu_mar_v2_vig_13rows_20260511.fits'
        self.arf_file_path = '/xifu/usr/share/sixte/instruments/new-athena-xifu/instdata/new_athena_xifu_mar_v2_no_filter.arf'
        self.rmf_file_path = '/xifu/usr/share/sixte/instruments/new-athena-xifu/instdata/new_athena_xifu_mar_v2_4eV_gaussian.rmf'
        
        self.PSF_image_path = '/xifu/home/mola/xifu_cluster_sim/data/PSF_image_NewAthena.fits'
        PSF_kernel = fits.getdata(self.PSF_image_path).astype(float)
        self.PSF_image = PSF_kernel/np.sum(PSF_kernel)
