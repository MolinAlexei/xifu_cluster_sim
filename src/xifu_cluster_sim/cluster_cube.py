# Basic stuff
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Astropy stuff
from astropy.cosmology import LambdaCDM
from astropy import units as units
from astropy import constants as const
from astropy.coordinates import SkyCoord, SkyOffsetFrame

# Jax stuff
import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True) ## NECESSARY FOR MY COMPUTATION OF THE NORM
import jax.random as random
import jax.numpy.fft as fft

from .grid import SpatialGrid3D, FourierGrid3D
from .xifu_config import XIFU_Config
from .turbulence import KolmogorovPowerSpectrum


class ClusterCube:
    """
    Define a cube containing all the properties of the simulated cluster
    """

    def __init__(self,
                 cluster_z = 0.1,
                 xifu_config = XIFU_Config(),
                 shape=(58, 58), 
                 los_factor=5,
                 x_off = 0, 
                 y_off = 0, 
                 z_off = 0):
        """
        Initialize the cluster cube with X, Y, Z coordinates

        Parameters:
            cluster_z (float): Redshift of cluster
            xifu_config (): Module with X-IFU instrument configuration
            shape (tuple): Shape of cluster cube along x,y
            los_factor (int): Shape ratio of cluster along z with respect to x,y
            x_off (int): Offset of cluster center along x
            y_off (int): Offset of cluster center along y
            z_off (int): Offset of cluster center along z


        """
        
        # Define cosmo to extract distances
        self.cosmo = LambdaCDM(H0 = 70, Om0 = 0.3, Ode0= 0.7)
        self.cluster_z = cluster_z
        self.ang_diam_distance_kpc = self.cosmo.angular_diameter_distance(cluster_z).to(units.kpc)

        # Properties of X-IFU
        self.xifu_config = xifu_config
    
        #self.pixsize_arcsec = (xifu_config.pixel_size_m/xifu_config.athena_focal_length * units.radian).to(units.arcsec)
        self.pixsize_kpc = (xifu_config.pixsize_arcsec * self.cosmo.kpc_proper_per_arcmin(cluster_z)).to(units.kpc)
        
        # Physical grid
        self.shape = shape
        self.grid = SpatialGrid3D(self.pixsize_kpc, shape, los_factor, x_off, y_off, z_off)
        
        # Fourier grid
        self.fourier_grid = FourierGrid3D(self.grid)
        
        # Get RA, Dec from physical grid
        # I ignore the contribution of Z, so I assume that the cluster is flat
        conversion = self.cosmo.kpc_proper_per_arcmin(cluster_z)
        coords = SkyCoord(
                            lon = -self.grid.Y * units.kpc / conversion,
                            lat = self.grid.X * units.kpc / conversion,
                            frame = SkyOffsetFrame(origin=SkyCoord(ra=0, dec=0, unit='deg')),
                            ).transform_to('icrs')
        
        # Extend along z-axis
        self.ra = coords.ra.to(units.deg).value
        self.ra = np.repeat(self.ra, repeats = shape[0]*los_factor, axis=2)
        
        self.dec = coords.dec.to(units.deg).value
        self.dec = np.repeat(self.dec, repeats = shape[0]*los_factor, axis =2)
       
    def create_density_cube(self, 
                            n0 = jnp.exp(-4.9), 
                            R_500 = 1309., 
                            r_c = jnp.exp(-2.7), 
                            gamma = 3, 
                            r_s = jnp.exp(-0.51), 
                            alpha = 0.7, 
                            beta = 0.39, 
                            eps = 2.6):
        r"""Compute the density within the cluster
        
        $$n_e^2(x)= n_0^2 \frac{(\frac{x}{r_c})^{-\alpha}}{(1 + (\frac{x}{r_c})^2)^{3\beta -\alpha /2}} \frac{1}{(1 + (\frac{x}{r_s})^{\gamma})^{\frac{\epsilon}{\gamma}}}$$

        Parameters:
            n0 (float): Density at core of cluster
            R_500 (float): Characteristic size of cluster
            r_c (float): Shape radius 1 in units of R/R500
            gamma (float): Shape factor outside of r_s
            r_s (float): Shape radius 2 in units of R/R500
            beta (float): Shape factor outside of r_c
            alpha (float): Shape factor at center of radius
            eps (float): Shape factor

        Returns:
            (jnp.array): Abundance function evaluated at the given radius
        """
        self.R_500 = R_500
        x = self.grid.R/R_500
        emiss = n0**2 * (x/r_c)**-alpha / (1 + (x/r_c)**2 )**(3*beta - alpha/2) / (1 + (x/r_s)**gamma)**(eps/gamma)
        #Cut emission at 5 R_500
        self.squared_density_cube = jnp.clip(emiss * jnp.heaviside(5. - x, 0), 0, 0.05**2) / 1.2

    def create_kT_cube(self,
                       M500 = 0.7,
                       T0 = 1.09,
                       rcool = jnp.exp(-4.4),
                       rt = 0.45,
                       TmT0 = 0.66,
                       acool = 1.33,
                       c2 = 0.3,
                       ):
        r"""Compute the temperature within the cluster.

        $$\dfrac{T(x)}{T_{500}} = T_0 \dfrac{\frac{T_\mathrm{min}}{T_0} + (\frac{x}{r_\mathrm{cool}})^{a_\mathrm{cool}}}{1 + (\frac{x}{r_\mathrm{cool}})^{a_\mathrm{cool}}} \frac{1}{(1 + (\frac{x}{r_t})^2)^{\frac{c}{2}}}$$

        Parameters:
            M500 (float): Characteristic mass of cluster
            rcool (float): Shape radius 1 in units of R/R500
            
        Returns:

        """

        # Scale T_500 to redshift
        Ez   = self.cosmo.efunc(self.cluster_z)
        T_500 = 8.85 * (M500*self.cosmo.h)**(2./3.) * Ez**(2./3.)

        # Switch to units of R_500
        x = self.grid.R/self.R_500

        
        term1 = (TmT0 + (x / rcool) ** acool)
        term2 = (1 + (x / rcool) ** acool) * (1 + (x / rt) ** 2) ** c2

        self.kT_cube = T_500 * T0 * term1 / term2 * jnp.heaviside(5. - x, 0)
        
    def create_Z_cube(self):
        """
        Compute the abundance function for a given radius, following Mernier et al 2017.
        All parameters are fixed.

        Parameters:
            
        Returns:
        
        """

        x = self.grid.R/self.R_500
        self.Z_cube = 0.21*(x+0.021)**(-0.48) - 6.54*jnp.exp( - (x+0.0816)**2 / (0.0027) )* jnp.heaviside(5. - x, 0)
                
        
    def create_norm_cube(self):
        """
        Compute the norm for the xspec apec model

        $$\text{norm} = \frac{1}{4 \pi (D_A(1+z))^2 \int n_e n_H dV}$$

        """
        
        ang_diam_distance_cm = self.ang_diam_distance_kpc.to(units.cm)
        
        xspec_normalization = 1e-14/(4*np.pi*(ang_diam_distance_cm*(1+self.cluster_z))**2)
        dV = (self.grid.pixsize.to(units.cm).value)**3
        ne_nH_dV = self.squared_density_cube * dV
        
        self.norm = xspec_normalization.value * ne_nH_dV
        
    def create_velocity_cube(self, 
                             rng_key,
                             sigma = 250., 
                             inj_scale = 300.,
                             dis_scale = 1e-3,
                             alpha = 11/3):
        """
        Creates a random realization of a velocity field

        Parameters:
            rng_key (jax.random.PRNGKey): Density at core of cluster
            sigma (float): Characteristic size of cluster
            inj_scale (float): Injection scale, in kiloparsecs
            dis_scale (float): Dissipation scale, in kiloparsecs
            alpha (float): Slope of the power spectrum

        """
        
        # I split the key because I have two realizations of a random gaussian field
        key_1, key_2 = jax.random.split(rng_key)
        
        # Power spectrum initialized
        self.power_spectrum = KolmogorovPowerSpectrum(sigma = sigma, 
                                                     inj_scale = inj_scale,
                                                     dis_scale = dis_scale,
                                                     alpha = alpha)
        
        self.K = self.fourier_grid.K.astype(np.float64)

        # Maximum inverse scale probed by discret grid
        k_max = jnp.max(self.fourier_grid.K.flatten()[1:])
                
        # Compute power spectrum over K and get normalization factor 
        PS_k, factor = self.power_spectrum(self.K, k_max / jnp.sqrt(2))

        # Classic generation of realization
        field_spatial = random.normal(key_1, shape=self.grid.shape)
        field_fourier = fft.rfftn(field_spatial)*jnp.sqrt(PS_k/self.grid.pixsize**3)
        field_spatial = fft.irfftn(field_fourier, s = self.grid.shape)

        # Add correction for small scales fluctuations
        field_spatial = field_spatial + random.normal(key_2, shape = self.grid.shape) * jnp.sqrt(factor)
        self.v_cube = field_spatial

    def convert_velocity_to_redshift(self):

        r"""
        Converts velocity to redshift following :

        $$z_mathrm{turb} = \sqrt{\frac{c + v}{c - v} - 1$$

        And 

        $$z_\mathrm{tot} = (1+z_mathrm{turb})(1+z_mathrm{cluster})$$


        """

        z_from_v = np.sqrt((const.c.value + self.v_cube*1e3)/(const.c.value - self.v_cube*1e3))-1

        self.z_cube = self.cluster_z + z_from_v + self.cluster_z * z_from_v
        
        
    def save_cube(self, path):
        """
        Saves the cubes of computed value of the cluster

        """

        np.savez_compressed(path,
                            X_coord_cube=self.grid.X,
                            Y_coord_cube=self.grid.Y,
                            Z_coord_cube=self.grid.Z,
                            ra_cube=self.ra,
                            dec_cube=self.dec,
                            ne_nH_cube=self.squared_density_cube,
                            norm_cube=self.norm,
                            kT_cube=self.kT_cube,
                            Z_cube=self.Z_cube,
                            v_cube=self.v_cube)

    def load_cube(self, path):
        """
        Loads the cubes of computed value of the cluster

        """
        
        data  = np.load(path)
        self.squared_density_cube = data['ne_nH_cube']
        self.norm = data['norm_cube']
        self.kT_cube = data['kT_cube']
        self.Z_cube = data['Z_cube']
        self.v_cube = data['v_cube']
        
    def create_input_maps(self,
                          binning, 
                          PSF_kernel,
                          save_maps = False,
                          path_save = './'):
        """
        Computes the input maps from the cluster cubes
        
        Parameters:
            binning (class): Binning used for the projection to maps
            PSF_kernel (jnp.array): PSF discretized on the pixel grid
            save_maps (bool): Whether to save the input maps in a npz file
            path_save (str): Path where to save the input maps

        """
        
        # Summing in each pixel
        summed_norm = jnp.sum(self.norm, axis = -1)
        summed_kT = jnp.sum(self.kT_cube*self.norm, axis = -1)
        summed_Z = jnp.sum(self.Z_cube*self.norm, axis = -1)
        summed_v = jnp.sum(self.v_cube*self.norm, axis = -1)
        summed_std = jnp.sum(self.v_cube**2*self.norm, axis = -1)

        # Convolution by PSF
        PSF_kernel*= 1/np.sum(PSF_kernel)
        summed_norm_conv = signal.convolve(summed_norm, PSF_kernel, mode = 'same')
        summed_kT_conv = signal.convolve(summed_kT, PSF_kernel, mode = 'same')
        summed_Z_conv = signal.convolve(summed_Z, PSF_kernel, mode = 'same')
        summed_v_conv = signal.convolve(summed_v, PSF_kernel, mode = 'same')
        summed_std_conv = signal.convolve(summed_std, PSF_kernel, mode = 'same')
        
        # Indices of which bin each pixel goes in
        bins_unique, inverse_indices = jnp.unique(binning.bin_num_pix, 
                                                  return_inverse=True, 
                                                  size = jnp.max(binning.bin_num_pix).astype(int))
        
        # Initialize vectors for summation in bins
        summed_norm_vec = jnp.zeros(binning.nb_bins)
        summed_kT_vec = jnp.zeros(binning.nb_bins)
        summed_Z_vec = jnp.zeros(binning.nb_bins)
        summed_v_vec = jnp.zeros(binning.nb_bins)
        summed_std_vec = jnp.zeros(binning.nb_bins)
        
        # We add to each bin the weighted sum of all pixels belonging to it
        summed_norm_vec = summed_norm_vec.at[inverse_indices].add(
                                                                summed_norm_conv[binning.X_pixels,binning.Y_pixels]
                                                                )
        summed_kT_vec = summed_kT_vec.at[inverse_indices].add(
                                                                summed_kT_conv[binning.X_pixels,binning.Y_pixels]
                                                                )
        summed_Z_vec = summed_Z_vec.at[inverse_indices].add(
                                                                summed_Z_conv[binning.X_pixels,binning.Y_pixels]
                                                                )
        summed_v_vec = summed_v_vec.at[inverse_indices].add(
                                                                summed_v_conv[binning.X_pixels,binning.Y_pixels]
                                                                )
        summed_std_vec = summed_std_vec.at[inverse_indices].add(
                                                                summed_std_conv[binning.X_pixels,binning.Y_pixels]
                                                                )
        
        # Divide by weighing (i.e. summed emission)
        weight_for_norm_vec = jnp.zeros(binning.nb_bins)
        weight_for_norm_vec = weight_for_norm_vec.at[inverse_indices].add(
                                                                            jnp.ones(self.shape)[binning.X_pixels,binning.Y_pixels]
                                                                        )
        summed_norm_vec_weighted = summed_norm_vec[inverse_indices] / weight_for_norm_vec[inverse_indices]
        summed_kT_vec_weighted = summed_kT_vec[inverse_indices]/summed_norm_vec[inverse_indices]
        summed_Z_vec_weighted = summed_Z_vec[inverse_indices]/summed_norm_vec[inverse_indices]
        summed_v_vec_weighted = summed_v_vec[inverse_indices]/summed_norm_vec[inverse_indices]
        summed_std_vec_weighted = jnp.sqrt(summed_std_vec[inverse_indices]/summed_norm_vec[inverse_indices] - summed_v_vec_weighted**2)


        
        # Create maps
        input_norm = jnp.zeros(self.shape)
        input_norm = input_norm.at[binning.X_pixels, binning.Y_pixels].set(summed_norm_vec_weighted)
        
        input_kT = jnp.zeros(self.shape)
        input_kT = input_kT.at[binning.X_pixels, binning.Y_pixels].set(summed_kT_vec_weighted)
        
        input_Z = jnp.zeros(self.shape)
        input_Z = input_Z.at[binning.X_pixels, binning.Y_pixels].set(summed_Z_vec_weighted)
    
        input_v = jnp.zeros(self.shape)
        input_v = input_v.at[binning.X_pixels, binning.Y_pixels].set(summed_v_vec_weighted)
        
        input_std = jnp.zeros(self.shape)
        input_std = input_std.at[binning.X_pixels, binning.Y_pixels].set(summed_std_vec_weighted)
        
        if save_maps:
            np.savez_compressed(path_save + 'input_maps.npz', 
                            input_norm=input_norm,
                            input_kT=input_kT,
                            input_Z=input_Z,
                            input_v=input_v,
                            input_std=input_std)
    
        return input_norm, input_kT, input_Z, input_v, input_std