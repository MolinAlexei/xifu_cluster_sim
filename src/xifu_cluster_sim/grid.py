from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import jax.numpy.fft as fft
import astropy.units as u


class SpatialGrid3D:

    def __init__(self, pixsize=8.71, shape=(58, 58), los_factor=5,
                    x_off = 0, y_off = 0, z_off = 0):
        """
        pixsize is in kpc 
        shape is the 2D shape of the image 
        crop_r_500 is the line of sight extent
        """

        self.pixsize = pixsize

        x_size, y_size = shape
        z_size = int(shape[0]*los_factor)

        self.x = jnp.linspace(self.pixsize * (-(x_size - 1)/2 + x_off), self.pixsize * ((x_size - 1)/2 + x_off), x_size) + pixsize/2
        self.y = jnp.linspace(self.pixsize * (-(y_size - 1)/2 + y_off), self.pixsize * ((y_size - 1)/2 + y_off), y_size) + pixsize/2
        self.z = jnp.linspace(self.pixsize * (-(z_size - 1)/2 + z_off), self.pixsize * ((z_size - 1)/2 + z_off), z_size)
        self.X, self.Y, self.Z = jnp.meshgrid(self.x, self.y, self.z, indexing='ij', sparse=True)
        self.y_i, self.x_i = jnp.indices(shape)
        self.R = jnp.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        self.shape = (x_size, y_size, z_size)
        self.volume = np.prod(self.shape)*pixsize**3

    

class FourierGrid3D:

    def __init__(self, spatial_grid):

        self.kx = fft.fftfreq(len(spatial_grid.x), d=spatial_grid.x[1] - spatial_grid.x[0])
        self.ky = fft.fftfreq(len(spatial_grid.y), d=spatial_grid.y[1] - spatial_grid.y[0])
        self.kz = fft.rfftfreq(len(spatial_grid.z), d=spatial_grid.z[1] - spatial_grid.z[0])
        KX, KY, KZ = jnp.meshgrid(self.kx, self.ky, self.kz, indexing='ij', sparse=True)
        self.K = jnp.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

        self.shape = self.K.shape
