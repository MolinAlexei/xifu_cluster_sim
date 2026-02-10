import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.cosmology import LambdaCDM
from astropy import units as units
import astropy.constants as const
from astropy.io import fits


from .cluster_cube import ClusterCube
from .photon_list import PhotonList, create_photon_list
from .events_management import merge_evt_files_recursive, create_count_map
from .run_sixte import run_sixte_sim_multiproc
from .xifu_config import XIFU_Config

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
                interp_files_path = '/xifu/home/mola/XIFU_Sims_Turbulence_NewConfig/Observation5/files/',
                xifu_config = XIFU_Config()
                ):

        self.sim_path = sim_path
        self.xifu_config = xifu_config

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
                background = True,
                astro_bkg_simput = '/xifu/home/mola/xifu_cluster_sim/data/background/astro_bkg.simput',
                cosmic_bkg_simput = '/xifu/home/mola/xifu_cluster_sim/data/background/cosmic_bkg.simput',
                skip_photon_list = False
                ):
        
        self.pointing_shape = self.xifu_config.pointing_shape

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

        if not skip_photon_list:
            ph_list = PhotonList(self.cluster,
                            E_bounds,
                            Z_scale,
                            T_scale, 
                            interp_spec_scale)

        print(x_offsets, y_offsets, pointing_names)

        for x_offset, y_offset, pointing_name in zip(x_offsets,
                                                    y_offsets,
                                                    pointing_names):
            pointing_path = self.sim_path + pointing_name
            if not os.path.exists(pointing_path):
                os.mkdir(pointing_path)


            if not skip_photon_list:
                print('Generating photon list')
                ph_list.select_sub_pointing(x_offset=x_offset,
                                            y_offset=y_offset,
                                            pointing_shape = self.pointing_shape)

                args = ph_list.get_args_list(numprocs = 8, 
                                            numthreads = 4)
                photonlist = create_photon_list(ph_list,
                                                args, 
                                                numproc = 8)

                print('Saving photon list')
                ph_list.write_photon_list_to_simputs(photonlist,
                                        num_divisions=32,
                                        path = pointing_path,
                                        name_format = 'ph_list')

            else :
                print('Skipping photon list')

            

            
            if x_offset < 0:
                ra_pointing = -x_offset * self.xifu_config.pixsize_degree.value
            else : 
                ra_pointing = 360 - x_offset * self.xifu_config.pixsize_degree.value
            dec_pointing = y_offset * self.xifu_config.pixsize_degree.value

            print('Runnning SIXTE')
            run_sixte_sim_multiproc(32,
                        path = pointing_path+'/sixte_files/',
                        ph_list_prefix="ph_list",
                        evt_file_prefix='events',
                        exposure=125e3,
                        ra = ra_pointing,
                        dec = dec_pointing,
                        std_xmlfile = self.xifu_config.std_xmlfile,
                        astro_bkg_simput = astro_bkg_simput, 
                        cosmic_bkg_simput = cosmic_bkg_simput, 
                        background = background)

            final_evt_file = pointing_path + '/sixte_files/events.fits'
            file_pattern = pointing_path + "/sixte_files/part[0-9]*/events_[0-9]*.fits"


            print('Merging events')
            merge_evt_files_recursive(final_evt_file, 
                        file_pattern, 
                        clobber=True)

            cnt_map_file = pointing_path + '/count_map.fits'
            create_count_map(final_evt_file, cnt_map_file, ra=ra_pointing, dec=dec_pointing)

    def merge_count_maps(self,
                          x_offsets,
                          y_offsets,
                          pointing_names,
                          exposure_times,
                          sim_path,
                          mosaic_name = '19p_mosaic',
                          show = False,
                          no_bkg = False):
    
    
        total_count_map = np.zeros(self.cluster.shape)

        for x,y,p_name,t_exp in zip(x_offsets, 
                                    y_offsets, 
                                    pointing_names,
                                    exposure_times):
            print('Adding:', int(x), int(y), 'with exposure:', int(t_exp))

            cnt_map = fits.getdata(self.sim_path + '{}/count_map.fits'.format(p_name))
            if no_bkg:
                cnt_map = fits.getdata(self.sim_path + '{}/count_map_no_bkg.fits'.format(p_name))

            s0,s1 = self.cluster.shape
            width, height = self.pointing_shape
            total_count_map[s0//2 - width//2 + y : s0//2 + width//2 + y,
                            s1//2 - height//2 + x : s1//2+ height//2 + x] += cnt_map

        if show:
            pixsize_arcmin = self.xifu_config.pixsize_degree.to(units.arcmin).value
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

        outdir_mosaic = self.sim_path+'/'+mosaic_name+'/'
        print('Path to save merge count map :', outdir_mosaic)

        pixsize_degree = self.xifu_config.pixsize_degree.value

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

    def merge_events(self,
                          x_offsets,
                          y_offsets,
                          pointing_names,
                          exposure_times,
                          sim_path,
                          mosaic_name = '19p_mosaic'):


                    
        for k,x,y,p_name,t_exp in zip(range(len(x_offsets)),
                                    x_offsets, 
                                    y_offsets, 
                                    pointing_names,
                                    exposure_times):

            print("Assembling event file of pointing %i" %(k+1))

            hdu = fits.open(sim_path + '{}/sixte_files/events.fits'.format(p_name) )      
            
            if k==0:
                ph_id     = np.max(hdu[1].data['PH_ID'])
                ph_id_min = np.min(hdu[1].data['PH_ID'])
                
                evt_list  = hdu[1].data
            else:
                hdu[1].data['PIXID'] = hdu[1].data['PIXID'] + k*self.xifu_config.xifu_pixel_number
                hdu[1].data['PH_ID'][hdu[1].data['PH_ID']<0] = hdu[1].data['PH_ID'][hdu[1].data['PH_ID']<0] + ph_id_min
                hdu[1].data['PH_ID'][hdu[1].data['PH_ID']>0] = hdu[1].data['PH_ID'][hdu[1].data['PH_ID']>0] + ph_id
                

                # I remove the offset of each pointing so that the vignetting only accounts for distance from the center
                # hdu[1].data['RA']  = hdu[1].data['RA']  + int(x)*self.xifu_config.pixsize_degree.value
                #hdu[1].data['DEC'] = hdu[1].data['DEC'] - int(y)*self.xifu_config.pixsize_degree.value

                ph_id     = np.max(hdu[1].data['PH_ID'])
                ph_id_min = np.min(hdu[1].data['PH_ID'])
                            
                evt_list  = np.hstack((evt_list, hdu[1].data))

        outdir_mosaic = sim_path+'/'+mosaic_name+'/'
                            
        outfile = outdir_mosaic + "/events.fits"
        
        hdu[1].data = evt_list
        hdu.writeto(outfile, overwrite = True)
