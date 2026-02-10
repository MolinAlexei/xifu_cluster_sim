import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
from astropy import units as units
import astropy.constants as const
from astropy.io import fits 

import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True) ## NECESSARY FOR MY COMPUTATION OF THE NORM

from joblib import Parallel, delayed
from tqdm.notebook import tqdm as tqdm_notebook
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import os

class PhotonList:
    """
    Create the photon lists associated 
    with the observation of a cluster, using
    an interpolation of spectra in temperature
    and abundance

    """
    def __init__(self, 
                 cluster, 
                 E_bounds, 
                 Z_scale, 
                 T_scale, 
                 interp_spec_table,
                 exposure = 125e3,
                 exposure_overhead = 10,
                 eff_area = 2e4,
                 fov_radius_pix = 26.5):
        """
        Load cluster object and define parameters of observation

        Parameters:
            cluster (object): cluster object
            E_bounds (np.array): Bounds in energy used to sample photons
            Z_scale (np.array) : Abundance scale used for interpolated grid
            T_scale (np.array) : Temperature scale used for interpolated grid
            interp_spec_table (np.array) : Table of spectra to interpolate from
            exposure (float) : Exposure in seconds
            exposure_overhead (float) : Multiplying factor for the exposure for oversampling of photon list
            eff_area (float) : Approximate total effective area in cm$^2$

        """
        self.cluster = cluster
        self.E_bounds = E_bounds
        self.E = (E_bounds[1:]+E_bounds[:-1])/2
        self.T_scale = T_scale
        self.Z_scale = Z_scale
        self.interp_spec_table = interp_spec_table
        self.exposure = exposure
        self.exposure_overhead = exposure_overhead
        self.nb_cells = np.prod(cluster.grid.shape)
        self.cluster_z = cluster.cluster_z
        self.eff_area = eff_area
        self.fov_radius_pix = fov_radius_pix
              
    def select_sub_pointing(self,
                           x_offset=0,
                           y_offset=0,
                           pointing_shape = (58,58)):
        
        total_shape = self.cluster.grid.shape
        X,Y = np.meshgrid(np.arange(total_shape[0]),
                          np.arange(total_shape[1]))
        
        X = X.astype('float64')
        Y = Y.astype('float64')
        mask = np.where(np.sqrt((X - total_shape[0]/2 - x_offset)**2 + (Y - total_shape[1]/2 - y_offset)**2) < self.fov_radius_pix)

        self.ra = self.cluster.ra[mask]
        self.dec = self.cluster.dec[mask]
        self.norm = self.cluster.norm[mask]
        self.kT_cube = self.cluster.kT_cube[mask]
        self.Z_cube = self.cluster.Z_cube[mask]
        self.z_cube = self.cluster.z_cube[mask]

    def process_los(self,ra_los,dec_los,norm_los,kT_los, Z_los, z_los):
        res = []
        for k in range(self.cluster.grid.shape[-1]):
            ra = ra_los[k]
            dec = dec_los[k]
            norm = norm_los[k]
            kT = kT_los[k]
            Z = Z_los[k]
            z = z_los[k]

            res.append(self.add_photons_interpTZ_new(ra, 
                                             dec, 
                                             norm, 
                                             kT, 
                                             Z, 
                                             z)
                      )

        return np.hstack([res_item for res_item in res if res_item is not None])
    
    
    def worker_block(self,args_block):
        # runs in one process, shares ph_list
        results = []
        with ThreadPoolExecutor(max_workers=self.numthreads) as executor:
            for r in executor.map(lambda args: self.process_los(*args), args_block):
                results.append(r)
        return results

    # Split your total args_list into 4 blocks
    def chunkify(self,lst, n):
        return [lst[i::n] for i in range(n)]
    
    def get_args_list(self,
                      numprocs= 4,
                      numthreads  = 8):
        self.numprocs = numprocs
        self.numthreads = numthreads
        los_len = self.cluster.grid.shape[-1]
        
        args_list = [
                        (ra.copy(), dec.copy(), norm.copy(), kT.copy(), Z.copy(), z.copy())
                        for ra, dec, norm, kT, Z, z in zip(
                            np.array(self.ra.reshape(-1,los_len)),
                            np.array(self.dec.reshape(-1,los_len)),
                            np.array(self.norm.reshape(-1,los_len)),
                            np.array(self.kT_cube.reshape(-1,los_len)),
                            np.array(self.Z_cube.reshape(-1,los_len)),
                            np.array(self.z_cube.reshape(-1,los_len))
                        )
                    ]
        return self.chunkify(args_list, numprocs)
        
        
    def create_photon_list_multiproc_old(self,
                                    numproc=None):
        '''
        Create the photon list associated with the loaded cluster object 

        Parameters:
            numproc (int): Number of available cores for the parallel computation
        '''   
        
        n_cells = np.prod(self.ra.shape)
        
        args_list = zip(
                        np.array(self.ra.flatten()[:n_cells]),
                        np.array(self.dec.flatten()[:n_cells]),
                        np.array(self.norm.flatten()[:n_cells]),
                        np.array(self.kT_cube.flatten()[:n_cells]),
                        np.array(self.Z_cube.flatten()[:n_cells]),
                        np.array(self.z_cube.flatten()[:n_cells])
                        )
        if numproc > 1 :
            batch_size = max(1, n_cells//(4*numproc))
        else :
            batch_size = 'auto'                                                       
        
        results = Parallel(n_jobs = numproc, 
                           backend = "loky",
                          batch_size = batch_size)(
                                delayed(self.add_photons_interpTZ_new)(
                                    ra,
                                    dec,
                                    norm,
                                    kT,
                                    Z,
                                    z
                                    ) for ra, dec, norm, kT, Z, z in tqdm_notebook(args_list, 
                                                                                   total = n_cells, 
                                                                                   desc = 'Creating photon list'
                                                                                )
                                )
                                                                
        
        results = [res_item for res_item in results if res_item is not None] 

        photon_list = np.hstack(results)
        self.fluxes = photon_list[0]
        self.energies = photon_list[1]
        self.ras = photon_list[2]
        self.decs = photon_list[3]
    
        
    def add_photons_interpTZ_new(self,
                                 ra,
                                 dec,
                                 norm,
                                 temperature,
                                 abundance,
                                 redshift):
        '''
        Utilitary function used to generate photons from a given point 
        in norm, temperature, abundance and redshift
        
        Parameters:
            ra (float) : RA of given point
            dec (float) : DEC of given point
            norm (float) : norm of given point
            temperature (float) : norm of given point
            abundance (float) : abundance of given point
            redshift (float) : redshift of given  point
        '''
        emin=self.E_bounds[:-1]
        emax=self.E_bounds[1:]
        #Compute number of photons for this cell
        indT=np.digitize(temperature,self.T_scale)-1
        indZ=np.digitize(abundance,self.Z_scale)-1

        # Interpolate from table
        dT = temperature - self.T_scale[indT]
        deltaT = self.T_scale[indT+1] - self.T_scale[indT]
        dZ = abundance - self.Z_scale[indZ]
        deltaZ = self.Z_scale[indZ+1] - self.Z_scale[indZ]
        delta_fluxT = self.interp_spec_table[indZ][indT+1] - self.interp_spec_table[indZ][indT]
        delta_fluxZ = self.interp_spec_table[indZ+1][indT] - self.interp_spec_table[indZ][indT]
        delta_fluxTZ = self.interp_spec_table[indZ][indT] + self.interp_spec_table[indZ+1][indT+1] - self.interp_spec_table[indZ][indT+1] - self.interp_spec_table[indZ+1][indT]    

        # Interpolated spectrum
        flux_inter=(delta_fluxT*dT/deltaT + delta_fluxZ*dZ/deltaZ + delta_fluxTZ*(dT/deltaT)*(dZ/deltaZ) + self.interp_spec_table[indZ][indT])*norm

        # Redshift
        E_shift = self.E*(1+self.cluster_z)/(1+redshift)
        flux_inter = np.interp(self.E, E_shift, flux_inter)

        # Flux in ergs.cm-2 for the simput file
        flux_erg = np.trapz(flux_inter*self.E)*units.keV.to(units.erg)

        # Flux of spectrum, in photons
        flux_spectrum = np.trapz(flux_inter)

        # Number of photons to draw
        nphotons = flux_spectrum * self.eff_area * self.exposure * self.exposure_overhead
        nphotons = np.random.poisson(nphotons)        
        
        # If there are any photons to draw
        if nphotons:

            # CDF of spectrum
            flux_cdf = np.cumsum(flux_inter/np.sum(flux_inter))

            # Single draw of random numbers
            random_numbers = np.random.uniform(size=2*nphotons)

            # Use first draw to get indices distributed according to CDF of spectrum
            search_indexes = np.searchsorted(flux_cdf,random_numbers[:nphotons]*flux_cdf[-1])

            # Use second draw to get energies and each is uniformly drawn in its energy bin
            energy_list = emin[search_indexes] + random_numbers[nphotons:]*(emax[search_indexes]-emin[search_indexes])
            
            # Make into arrays
            flux_erg_list = flux_erg*np.ones(nphotons)/nphotons
            ra_list = ra*np.ones(nphotons)
            dec_list = dec*np.ones(nphotons)

            return np.vstack((flux_erg_list, energy_list, ra_list, dec_list))

        else : 
            return None

    def write_photon_list_to_simputs(self,
                                    photon_list,
                                    num_divisions=None,
                                    path = '/xifu/home/mola/xifu_cluster_sim/',
                                    name_format = 'ph_list'):
        '''
        Divide the photon list into equal divisions, 
        and write each to a simput file.
        Parameters
            num_divisions (float) : Number of divisions of the photon list
        '''
        self.fluxes = photon_list[0]
        self.energies = photon_list[1]
        self.ras = photon_list[2]
        self.decs = photon_list[3]
        
        # Get indices for equal divisions
        indexes_phlist = np.linspace(0,self.ras.shape[0]-1, num_divisions+1).astype(int)
        
        if not os.path.exists(path + '/sixte_files/'):
            os.mkdir(path + '/sixte_files/')
        
        # Iterate for number of divisions
        for i in range(num_divisions):
            
            
            
            # Create sub folder for each simput
            if not os.path.exists(path + '/sixte_files/part{}/'.format(i)):
                os.mkdir(path + '/sixte_files/part{}/'.format(i))

            # Write simput
            self.write_single_simput_file(
                                        self.ras[indexes_phlist[i]:indexes_phlist[i+1]], 
                                        self.decs[indexes_phlist[i]:indexes_phlist[i+1]], 
                                        self.energies[indexes_phlist[i]:indexes_phlist[i+1]], 
                                        np.sum(self.fluxes[indexes_phlist[i]:indexes_phlist[i+1]]), 
                                        path + '/sixte_files/part{}/'.format(i) + name_format +'_{}.fits'.format(i), 
                                        clobber=True
                                        )

    def write_single_simput_file(self,
                        ra, 
                        dec, 
                        energy, 
                        flux, 
                        simputfile, 
                        clobber=False):
        """
        
        Write photons to a SIMPUT file that may be read by the SIXTE instrument
        simulator. See the SIMPUT definition for reference: 
        http://hea-www.harvard.edu/heasarc/formats/simput-1.1.0.pdf
        
        Parameters
            ra (np.array) : The RA positions of the photons in degrees.
            dec (np.array) : The Dec positions of the photons in degrees.
            energy (np.array) : The energies of the photons in keV. 
            flux (float) : Total flux of photons in erg/s/cm**2 in the reference energy band.
            simputfile (str) : Name of the simput file to write
            clobber (bool) : Set to True to overwrite previous files.
        """


        col1 = fits.Column(name='ENERGY', format='E', array=energy)
        col2 = fits.Column(name='RA', format='D', array=ra)
        col3 = fits.Column(name='DEC', format='D', array=dec)

        coldefs = fits.ColDefs([col1, col2, col3])
        
        #tbhdu=fits.new_table(coldefs)
        tbhdu = fits.BinTableHDU.from_columns(coldefs)
        tbhdu.name = "PHLIST"

        tbhdu.header["HDUCLASS"] = "HEASARC/SIMPUT"
        tbhdu.header["HDUCLAS1"] = "PHOTONS"
        tbhdu.header["HDUVERS"] = "1.1.0"
        tbhdu.header["EXTVER"] = 1
        tbhdu.header["REFRA"] = 0.0
        tbhdu.header["REFDEC"] = 0.0
        tbhdu.header["TUNIT1"] = "keV"
        tbhdu.header["TUNIT2"] = "deg"
        tbhdu.header["TUNIT3"] = "deg"
        
        #Construct source catalog    
        col1 = fits.Column(name='SRC_ID', format='J', array=np.array([1]).astype("int32"))
        col2 = fits.Column(name='RA', format='D', array=np.array([0.0]))
        col3 = fits.Column(name='DEC', format='D', array=np.array([0.0]))
        col4 = fits.Column(name='E_MIN', format='D', array=np.array([float(self.E_bounds[0])]))
        col5 = fits.Column(name='E_MAX', format='D', array=np.array([float(self.E_bounds[-1])]))
        col6 = fits.Column(name='FLUX', format='D', array=np.array([flux]))
        col7 = fits.Column(name='SPECTRUM', format='80A', 
                             array=np.array(["[PHLIST,1]"]))
        col8 = fits.Column(name='IMAGE', format='80A', 
                             array=np.array(["[PHLIST,1]"]))
                            
        coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8])
            
        #wrhdu=fits.new_table(coldefs)
        wrhdu = fits.BinTableHDU.from_columns(coldefs)
        wrhdu.name = "SRC_CAT"
                                    
        wrhdu.header["HDUCLASS"] = "HEASARC"
        wrhdu.header["HDUCLAS1"] = "SIMPUT"
        wrhdu.header["HDUCLAS2"] = "SRC_CAT"
        wrhdu.header["HDUVERS"] = "1.1.0"
        wrhdu.header["RADECSYS"] = "FK5"
        wrhdu.header["EQUINOX"] = 2000.0
        wrhdu.header["TUNIT2"] = "deg"
        wrhdu.header["TUNIT3"] = "deg"
        wrhdu.header["TUNIT4"] = "keV"
        wrhdu.header["TUNIT5"] = "keV"
        wrhdu.header["TUNIT6"] = "erg/s/cm**2"

        primary_hdu = fits.PrimaryHDU()
        thdulist = fits.HDUList([primary_hdu, wrhdu,tbhdu])
        thdulist.writeto(simputfile,overwrite=clobber)


def create_photon_list(ph_list_object,
                      blocks,
                      numproc, 
                      ):
    with Pool(processes=numproc, initializer=lambda: globals().update({'ph_list': ph_list_object})) as pool:
        results_nested = []
        for block_result in pool.imap(ph_list_object.worker_block, blocks):
            results_nested.append(block_result)
            #results_nested = pool.map(worker_block, tqdm(blocks, desc='Processes'))

        results = [r for block in results_nested for r in block]
    
    return np.hstack(results)