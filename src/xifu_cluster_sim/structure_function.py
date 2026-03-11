import numpy as np 
from itertools import combinations
import scipy.stats as stats
import jax.numpy as jnp

import time

class StructureFunction:
    r"""
    Compute the second order structure function on a 2D binned map as
    $\mathrm{SF}(s) = \frac{1}{N_p(s)} \sum_{d(\mathcal{W}_1, \mathcal{W}_2) = s} |C_{\mathcal{W}_1} - C_{\mathcal{W}_2}|^2$

    with 
    $\mathcal{W}_1, \mathcal{W}_2$ two separate regions, $d$ the distance that separates them and $N_p$ the number of pairs separated by $s$.
    """

    def __init__(self,
                bins = np.geomspace(1,40,15)
                ):
        """
        Initialize

        Parameters:
            bins (np.array): Bins over which the structure function is computed, in units of pixels.
        """

        self.bins = bins

    def compute_from_vector(self,
                        binning,
                        v_bin_vec):
        """
        Computes the 2nd order structure function from a vector.
        This vector is taken from a binning, where the i-th value of the vector is the value in the i-th bin of the map.
        
        Parameters:
            binning (binning): Binning instance
            v_bin_vec (jnp.array): Array of the value in each bin
        

        Returns:
            bin_dists (jnp.array): Separations of the SF
            bin_means (jnp.array): Values of the SF
        """
        
        #Indexes of all possible combinations
        idx = jnp.triu_indices(len(binning.xBar_bins), k = 1)
        
        #Vector of all possible separations
        sep_vec = jnp.hypot(binning.xBar_bins[idx[0]] - binning.xBar_bins[idx[1]],
                            binning.yBar_bins[idx[0]] - binning.yBar_bins[idx[1]])
        
        #Vector of (v_i - v_j)^2 
        v_vec = (v_bin_vec[idx[0]] - v_bin_vec[idx[1]])**2
        
        ### PREVIOUS VERSION
        #Now compute binned statistics
        #Average SF in each bin
        #bin_means, bin_edges, binnumber = stats.binned_statistic(sep_vec, v_vec, 'mean', bins = self.bins)
        #Average distance in each bin
        #bin_dists, bin_edges, binnumber = stats.binned_statistic(sep_vec, sep_vec, 'mean', bins = self.bins)

        ### JAX VERSION
        hist_weighted, bin_edges = jnp.histogram(sep_vec, bins = self.bins, weights = v_vec)
        hist_numbers, bin_edges = jnp.histogram(sep_vec, bins = self.bins)
        hist_dists, _ = jnp.histogram(sep_vec, bins = self.bins, weights = sep_vec)

        #Average SF in each bin
        bin_means = hist_weighted/hist_numbers

        #Average distance in each bin
        hist_dists = hist_dists/hist_numbers

        
        #Note : in principle, we shouldn't be taking the average distance in each bin, we should
        #take the bin center, but that's for comparison purposes with Edo's code
        #For most cases, with a lot of values in each bin, the average is equivalent with the bin center.
        
        return hist_dists, bin_means

    def compute_from_map(self, binning, v_map):
        """
        Computes the 2nd order structure function of a binned image with arbitrary binning.
        
        Parameters:
            binning (binning): Binning instance
            v_bin_vec (jnp.array): Map of the binned values
        
        Returns:
            bin_dists (jnp.array): Separations of the SF
            bin_means (jnp.array): Values of the SF
        """
        
        v_bin_vec = jnp.ones(binning.nb_bins)
        v_bin_vec = v_bin_vec.at[binning.bin_num_pix].set(v_map[binning.X_pixels,
                                                                binning.Y_pixels])

        #Indexes of all possible combinations
        idx = jnp.triu_indices(binning.nb_bins, k = 1)
        
        #Vector of all possible separations
        sep_vec = jnp.hypot(binning.xBar_bins[idx[0]] - binning.xBar_bins[idx[1]],
                            binning.yBar_bins[idx[0]] - binning.yBar_bins[idx[1]])

        #Vector of (v_i - v_j)^2 
        v_vec = (v_bin_vec[idx[0]] - v_bin_vec[idx[1]])**2
        
        ### PREVIOUS VERSION
        #Now compute binned statistics
        #Average SF in each bin
        #bin_means, bin_edges, binnumber = stats.binned_statistic(sep_vec, v_vec, 'mean', bins = self.bins)
        #Average distance in each bin
        #bin_dists, bin_edges, binnumber = stats.binned_statistic(sep_vec, sep_vec, 'mean', bins = self.bins)

        ### JAX VERSION
        hist_weighted, bin_edges = jnp.histogram(sep_vec, bins = self.bins, weights = v_vec)
        hist_numbers, bin_edges = jnp.histogram(sep_vec, bins = self.bins)
        hist_dists, _ = jnp.histogram(sep_vec, bins = self.bins, weights = sep_vec)

        #Average SF in each bin
        bin_means = hist_weighted/hist_numbers

        #Average distance in each bin
        hist_dists = hist_dists/hist_numbers

        
        #Note : in principle, we shouldn't be taking the average distance in each bin, we should
        #take the bin center, but that's for comparison purposes with Edo's code
        
        return hist_dists, bin_means
