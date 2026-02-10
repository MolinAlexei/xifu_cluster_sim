# instrument_config.py
from dataclasses import dataclass

import astropy.units as units
import astropy.constants as const

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
    def __init__(self):
        self.pixel_size_m = 317e-6  #meters
        self.athena_focal_length = 12.0 #meters
        self.pointing_shape = (58,58)

        self.pixsize_arcsec = (self.pixel_size_m / self.athena_focal_length * units.radian).to(units.arcsec)
        self.pixsize_degree = (self.pixel_size_m / self.athena_focal_length * units.radian).to(units.degree)

        self.xifu_pixel_number = 1504

        self.std_xmlfile = '/xifu/usr/share/sixte/instruments/athena-xifu_2024_11/baseline/xifu_nofilt_infoc_high_bkg_AM.xml'