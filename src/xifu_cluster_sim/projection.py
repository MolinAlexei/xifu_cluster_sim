import jax.numpy as jnp
from jax.numpy import fft as fft


def my_convolution(array1, array2):
    # My implementation of convolution of 2d arrays on calmip 
    # because of issues with CuDNN if I try to use regular convolve

    # Convolution with FFTs
    convolution = jnp.real(
                        fft.fftshift(
                            fft.ifftn(fft.fftn(array1) * jnp.conj(fft.fftn(array2)))
                                    )
                          )

    # Reshape convolution to match output of regular scipy convolve
    return jnp.roll(convolution,-1, axis = (0,1))


class VCubeProjection_v2 :
    """
    Projection of velocity cube with emission weighting in binned map
    """
    def __init__(self, binning, em_cube, PSF_kernel):
        """
        Initialize

        Parameters:
            binning (jnp.Module): Binning
            em_cube (jnp.array): 3D emissivity cube
            PSF_kernel (jnp.array): PSF discretized on the pixel grid
        """
        
        self.binning = binning
        self.em = em_cube
        self.PSF_kernel = PSF_kernel
        
    def __call__(self,v):
        
        """
        Uses counts to weight and project the velocity cube
        into a centroid shift map and broadening map.
        Uses a convolution with product in Fourier space.
        
        Parameters:
            v (jnp.array): Velocity cube

        Returns:
            v_map (jnp.array): Binned image of the emission weighted centroid shift
            std_map (jnp.array): Binned image of the emission weighted broadening
            binned_v_weighted (jnp.array): Vector of emission weighted centroid shift in each bin
            binned_std_weighted (jnp.array): Vector of emission weighted broadening in each bin
            
        """
        
        # This is a fully jaxified version
    
        # Summing emission weighted speed in each pixel        
        v_vec = jnp.sum(v*self.em, axis = -1)
        std_vec = jnp.sum(v**2*self.em, axis = -1)
        count_vec = jnp.sum(self.em, axis = -1)
        
        # Convolution by PSF
        v_vec_conv = my_convolution(v_vec, self.PSF_kernel)
        std_vec_conv = my_convolution(std_vec, self.PSF_kernel)
        count_vec_conv = my_convolution(count_vec, self.PSF_kernel)
        
        # Extracting values within pixels from matrix
        v_vec_conv_pix = v_vec_conv[self.binning.X_pixels,self.binning.Y_pixels]
        std_vec_conv_pix = std_vec_conv[self.binning.X_pixels,self.binning.Y_pixels]
        count_vec_conv_pix = count_vec_conv[self.binning.X_pixels,self.binning.Y_pixels]


        # Indices of which bin each pixel goes in
        bins_unique, inverse_indices = jnp.unique(self.binning.bin_num_pix, 
                                                  return_inverse=True, 
                                                  size = self.binning.nb_bins)

        # Binned vectors
        bin_v = jnp.zeros(len(bins_unique))
        bin_counts = jnp.zeros(len(bins_unique))
        bin_std = jnp.zeros(len(bins_unique))

        # We add to each bin the weighted sum of all pixels belonging to it
        bin_v = bin_v.at[inverse_indices].add(v_vec_conv_pix)
        bin_counts = bin_counts.at[inverse_indices].add(count_vec_conv_pix)
        bin_std = bin_std.at[inverse_indices].add(std_vec_conv_pix)


        # Divide by weighing (i.e. summed emission)
        binned_v_weighted = bin_v / bin_counts
        binned_std_weighted = jnp.sqrt((bin_std/bin_counts - binned_v_weighted **2))

        # Create maps
        v_map = jnp.zeros(self.binning.shape)
        v_map = v_map.at[self.binning.X_pixels, self.binning.Y_pixels].set(binned_v_weighted[inverse_indices])

        std_map = jnp.zeros(self.binning.shape)
        std_map = std_map.at[self.binning.X_pixels, self.binning.Y_pixels].set(binned_std_weighted[inverse_indices])

        # Return weighted and binned values
        return v_map, std_map, binned_v_weighted, binned_std_weighted