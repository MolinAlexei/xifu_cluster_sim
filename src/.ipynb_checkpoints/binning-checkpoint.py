import jax.numpy as jnp
import numpy as np
import pickle
from astropy.io import fits
from vorbin.voronoi_2d_binning import *

class LoadBinning : 
    """
    Get the different arrays used for a simulation instance from a pickle 
    binning as created by xifusim. The count map is needed for returning the
    count_weighted barycentre of each bin.
    """
    
    def __init__(self,
                 shape = (360,360),
                 binning_file = '/xifu/home/mola/Turbu_300kpc_mosaics/repeat10_125ks/19p_region_200/region_files/19p_region_dict.p',
                 count_map_file = '/xifu/home/mola/Turbu_300kpc_mosaics/repeat10_125ks/19p_count_image.fits'):
        
        self.shape = shape
        self.binning_dict, self.region_image = pickle.load(open(binning_file, 'rb'), encoding="bytes")
        self.countmap = jnp.array(fits.getdata(count_map_file), dtype = 'float32')

    def __call__(self):
        
        # Load pickle
        
        # Load count map
        
        # Initialize empty lists
        binnum = []
        X = []
        Y = []
        xBar = []
        yBar = []
        
        # Iterate on the number of bins
        for k in range(len(self.binning_dict)-1):
            
            binnum.extend([k] * len(self.binning_dict[k][0]) )
            x = self.binning_dict[k][1][0]
            y = self.binning_dict[k][1][1]
            X.extend(x)
            Y.extend(y)
            #Count weighted average for the barycenter of the bin
            xBar.append(jnp.average(jnp.array(x), weights = self.countmap[x,y]))
            yBar.append(jnp.average(jnp.array(y), weights = self.countmap[x,y]))
        
        # Arrays of x and y coordinate of each pixel on the xifusim images
        self.X_pixels = jnp.array(X)
        self.Y_pixels = jnp.array(Y)
        
        # Array of the bin number of each pixel
        self.bin_num_pix = jnp.array(binnum)
        
        # Number of bins
        self.nb_bins = len(jnp.unique(jnp.array(binnum)))
        
        # Arrays of the count-weighted barycenters
        self.xBar_bins = jnp.array(xBar)
        self.yBar_bins = jnp.array(yBar)
        
        # Map of the bin numbers (mainly used as a sanity check)
        self.bin_nb_map = jnp.zeros(self.shape)
        self.bin_nb_map = self.bin_nb_map.at[self.X_pixels, self.Y_pixels].set(self.bin_num_pix)
        
        return self.X_pixels, self.Y_pixels, self.bin_num_pix, self.nb_bins, self.xBar_bins, self.yBar_bins, self.bin_nb_map


    
class MakeBinning():
    """
    Make a Voronoi binning of the count map
    """
    def __init__(self,
                 shape = (360,360),
                 binning_file = '/xifu/home/mola/Turbu_300kpc_mosaics/repeat10_125ks/19p_region_200/region_files/19p_region_dict.p',
                 count_map_file = '/xifu/home/mola/Turbu_300kpc_mosaics/repeat10_125ks/19p_count_image.fits'):
        
        self.shape = shape
        self.binning_file = binning_file
        self.countmap = jnp.array(fits.getdata(count_map_file), dtype = 'float32')

    def compute_voronoi_binmap(self,
                               save_dict = False,
                               pix2xy_file = None,
                               snr = 200):
        '''
        Computes the dictionnary for a voronoi region

        - image (array) containing the equivalent fits file
        - mask_file (array) containing the mask file
        - voronoi_output (string) points to output txt file
        - binmap (string) points to output pickle file
        - snr (int) S/N ratio targeted
        '''
        
        # Arrays of x and y coordinate of each pixel on the xifusim images
        X, Y = np.where(self.countmap != 0. )
        self.X_pixels = X
        self.Y_pixels = Y
        
        # Voronoi binning
        binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(X, 
                                                                                  Y, 
                                                                                  self.countmap[X,Y], 
                                                                                  np.sqrt(self.countmap)[X,Y], 
                                                                                  snr, 
                                                                                  plot = 0, 
                                                                                  pixelsize = 1., 
                                                                                  quiet = 0)

        
        # Array of the bin number of each pixel
        self.bin_num_pix = jnp.array(binNum)
        
        # Number of bins
        self.nb_bins = len(jnp.unique(jnp.array(binNum)))
        
        # Arrays of the count-weighted barycenters
        self.xBar_bins = jnp.array(xBar)
        self.yBar_bins = jnp.array(yBar)
        
        # Map of the bin numbers (mainly used as a sanity check)
        self.bin_nb_map = jnp.zeros(self.shape)
        self.bin_nb_map = self.bin_nb_map.at[self.X_pixels, self.Y_pixels].set(self.bin_num_pix)
        
        # Create dict to be saved for consistency with old sims
        X_list = [self.X_pixels[np.where(self.bin_num_pix == k)] for k in range(self.nb_bins)]
        Y_list = [self.Y_pixels[np.where(self.bin_num_pix == k)] for k in range(self.nb_bins)]
        img_pix_numbers = np.zeros(self.shape, dtype = 'int32')
        
        pixid2xy_dict = pickle.load(open(pix2xy_file,'rb'),encoding="bytes")
        for key, value in pixid2xy_dict.items():
            # This is just a trick for the dict making below
            img_pix_numbers[value] = key

        # This structure is consistent with the old sims (but we can argue its not ideal)
        binning_dict = dict(
                         zip(
                            np.arange(0,self.nb_bins+1), 
                            [
                                [img_pix_numbers[y,x].tolist(), 
                                (x.tolist(),
                                 y.tolist()
                                )
                             ] for x,y in zip(X_list, Y_list)
                            ]
                            )
                        )
        self.binning_dict = binning_dict
        
        if save_dict:
            pickle.dump((self.binning_dict,self.bin_nb_map),
                    open(self.binning_file,'wb'))
        
        return self.X_pixels, self.Y_pixels, self.bin_num_pix, self.nb_bins, self.xBar_bins, self.yBar_bins, self.bin_nb_map
    
        
        