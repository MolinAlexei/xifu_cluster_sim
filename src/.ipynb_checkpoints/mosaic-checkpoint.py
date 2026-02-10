import numpy as np
import jax.numpy as jnp
from astropy.cosmology import LambdaCDM
from astropy import units as units
import astropy.constants as const
from astropy.io import fits

from grid import SpatialGrid3D, FourierGrid3D
from turbulence import KolmogorovPowerSpectrum
from cluster_cube import ClusterCube
from binning import LoadBinning, MakeBinning
from photon_list import PhotonList, create_photon_list
from events_management import merge_evt_files_recursive, create_count_map
from run_sixte import run_sixte_sim_multiproc
from spectra import MakeSpectra
from spectral_fitting import FitSpectra

import jax.random as random

import os

def rng_key():

    return random.PRNGKey(np.random.randint(0, int(1e6)))

class Mosaic:
    def __init__(self,
                sim_path,
                cluster = None,
                binning_file = None,
                mosaic_shape = (232,232),
                velocity_file = None,
                interp_files_path = '/xifu/home/mola/XIFU_Sims_Turbulence_NewConfig/Observation5/files/'
                ):

        self.sim_path = sim_path
        

        if cluster is None :

            cluster = ClusterCube(shape = mosaic_shape)
            cluster.create_density_cube()
            cluster.create_norm_cube()
            cluster.create_kT_cube()
            cluster.create_Z_cube()
            if velocity_file is None :
                cluster.create_velocity_cube(rng_key())
            else :
                v_cube = jnp.load(velocity_file)
                ### Reshape the cube
                s0,s1,s2 = v_cube.shape

                # Extract required portion
                v_cube_extract = v_cube[int(s0/2 - cluster.grid.shape[-1]/2):int(s0/2 + cluster.grid.shape[-1]/2),
                                       int(s1/2- cluster.grid.shape[0]/2) : int(s1/2+ cluster.grid.shape[0]/2),
                                       int(s2/2- cluster.grid.shape[1]/2) : int(s2/2+ cluster.grid.shape[1]/2)]

                # Swap axes as X is l.o.s for me, and Z is l.o.s. in old cluster code
                v_cube_extract = jnp.swapaxes(v_cube_extract, 0,2)

                # Flip on axis (for some reason, the old code flipped one axis...)
                v_cube_extract = jnp.flip(v_cube_extract, axis = 1)

                # Assign to cluster
                cluster.v_cube = jnp.roll(v_cube_extract, -1, axis = 1)

                cluster.convert_velocity_to_redshift()


        self.cluster = cluster

        self.binning_file = binning_file
        self.interp_files_path = interp_files_path

    def run_mosaic(self,
                x_offsets,
                y_offsets,
                pointing_names,
                pointing_shape = (58,58)):
        
        self.pointing_shape = pointing_shape

        # Paths to the file containing interpolated fluxes and spectra
        print('Loading interpolation tables')
        cluster_redshift = self.cluster.cluster_z
        flux_file = self.interp_files_path + "/flux_interp_z%.1f.npy" %(cluster_redshift)
        flux_interp = self.interp_files_path + "/flux_table_z%.1f.npy" %(cluster_redshift)

        interp_spec_scale = np.load(flux_interp,allow_pickle=True)
        Z_scale,T_scale,_,_ = np.load(flux_file,allow_pickle=True)

        # Energy bounds (hardcoded but I do not expect them to change)
        E_low = 0.2
        E_high = 12.
        E_step = 4e-4
        E_n_steps = int((E_high-E_low)/E_step + 1)
        E_bounds = np.linspace(E_low,
                               E_high, 
                               E_n_steps) 

        ph_list = PhotonList(self.cluster,
                        E_bounds,
                        Z_scale,
                        T_scale, 
                        interp_spec_scale)

        print(x_offsets, y_offsets, pointing_names)

        for x_offset, y_offset, pointing_name in zip(x_offsets,
                                                    y_offsets,
                                                    pointing_names):

            print('Generating photon list')
            ph_list.select_sub_pointing(x_offset=x_offset,
                                        y_offset=y_offset,
                                        pointing_shape = pointing_shape)

            args = ph_list.get_args_list(numprocs = 8, 
                                        numthreads = 4)
            photonlist = create_photon_list(ph_list,
                                            args, 
                                            numproc = 8)

            pointing_path = self.sim_path + pointing_name
            if not os.path.exists(pointing_path):
                os.mkdir(pointing_path)

            print('Saving photon list')
            ph_list.write_photon_list_to_simputs(photonlist,
                                    num_divisions=32,
                                    path = pointing_path,
                                    name_format = 'ph_list')

            if x_offset < 0:
                ra_pointing = -x_offset * self.cluster.pixsize_arcsec.to(units.degree).value
            else : 
                ra_pointing = 360 - x_offset * self.cluster.pixsize_arcsec.to(units.degree).value
            dec_pointing = y_offset * self.cluster.pixsize_arcsec.to(units.degree).value

            print('Runnning SIXTE')
            run_sixte_sim_multiproc(32,
                        path = pointing_path+'/sixte_files/',
                        ph_list_prefix="ph_list",
                        evt_file_prefix='events',
                        exposure=125e3,
                        ra = ra_pointing,
                        dec = dec_pointing,
                        std_xmlfile='/xifu/usr/share/sixte/instruments/athena-xifu_2024_11/baseline/xifu_nofilt_infoc.xml',
                        astro_bkg_simput=None, 
                        cosmic_bkg_simput=None, 
                        background=True)

            final_evt_file = pointing_path + '/sixte_files/events.fits'
            file_pattern = pointing_path + "/sixte_files/part[0-9]*/events_[0-9]*.fits"


            print('Merging events')
            merge_evt_files_recursive(final_evt_file, 
                        file_pattern, 
                        clobber=True)

            cnt_map_file = pointing_path + '/count_map.fits'
            create_count_map(final_evt_file, cnt_map_file, ra=ra_pointing, dec=dec_pointing)

    def create_map_mosaic(self,
                          x_offsets,
                          y_offsets,
                          pointing_names,
                          exposure_times,
                          sim_path,
                          mosaic_name = '19p_mosaic',
                          athena_focal_length_m = 12.,
                          show = False,
                          xifu_pixel_size_m = 317e-6):
    
    
        total_count_map = np.zeros(self.cluster.shape)

        for x,y,p_name,t_exp in zip(x_offsets, 
                                    y_offsets, 
                                    pointing_names,
                                    exposure_times):
            print('Adding:', int(x), int(y), 'with exposure:', int(t_exp))

            cnt_map = fits.getdata(self.sim_path + '{}/count_map.fits'.format(p_name))

            s0,s1 = self.cluster.shape
            width, height = self.pointing_shape
            total_count_map[s0//2 - width//2 + y : s0//2 + width//2 + y,
                            s1//2 - height//2 + x : s1//2+ height//2 + x] += cnt_map

        if show:
            pixsize_arcmin = (xifu_pixel_size_m/athena_focal_length_m * units.radian).to(units.arcmin).value
            size = total_count_map.shape[0]//2

            plt.imshow(total_count_map, 
                       extent=(-size*pixsize_arcmin, 
                               size*pixsize_arcmin,
                                -size*pixsize_arcmin, 
                                size*pixsize_arcmin), 
                       origin='lower', 
                       cmap='inferno',
                       interpolation='none',
                       norm=LogNorm())

            plt.xlabel("RA (arcmin)") #, **axis_font)
            plt.ylabel("DEC (arcmin)") #, **axis_font)
            colorbar=plt.colorbar()
            plt.title('{} count map'.format(mosaic_name))
            plt.show()

        outdir_mosaic = slef.sim_path+'/'+mosaic_name+'/'
        print('Path to save merge count map :', outdir_mosaic)

        pixsize_degree = (xifu_pixel_size_m/athena_focal_length_m * units.radian).to(units.degree).value

        hdu = fits.PrimaryHDU(total_count_map)
        outfile_header = hdu.header
        outfile_header['EXTEND']  = True
        outfile_header['WCSAXES'] = 2
        outfile_header['CRPIX1']  = total_count_map.shape[0]//2
        outfile_header['CRPIX2']  = total_count_map.shape[1]//2
        outfile_header['CDELT1']  = -pixsize_degree
        outfile_header['CDELT2']  = pixsize_degree
        outfile_header['CUNIT1']  = 'deg     '
        outfile_header['CUNIT2']  = 'deg     '
        outfile_header['CTYPE1']  = 'RA---TAN'
        outfile_header['CTYPE2']  = 'DEC--TAN'
        outfile_header['CRVAL1']  = 0.0
        outfile_header['CRVAL2']  = 0.0
        outfile_header['LONPOLE'] = 180.0
        outfile_header['LATPOLE'] =  0.0
        outfile_header['RADESYS'] = 'ICRS    '
        hdulist = fits.HDUList([hdu])
        if not os.path.exists(outdir_mosaic):
            os.mkdir(outdir_mosaic)
        if not os.path.exists(outdir_mosaic+'count_map.fits'):
            print('fits written out!')
        elif os.path.exists(outdir_mosaic+'count_map.fits'):
            print('fits overwritten!')
        hdulist.writeto(outdir_mosaic+'count_map.fits',overwrite=True)

        return total_count_map