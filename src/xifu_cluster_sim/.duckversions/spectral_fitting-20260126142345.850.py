import numpy as np
from astropy import units as units
import astropy.constants as const
from astropy.io import fits
import xspec as xs
import re as re

from joblib import Parallel, delayed
import joblib
from tqdm.notebook import tqdm as tqdm_notebook

from .xifu_config import XIFU_Config


def load_model(infile, pars_only=False):
    '''
    Retourne le model defini dans le fichier infile
    - infile (string) .xcm file
    - pars_only (boolean) option to return paramaters only
    '''
    #On recupere le nom du modele
    model_file = open(infile,'r')
    line = model_file.readline()
    while 'model' not in line:
        if 'cosmo' in line:
            line = line.replace('\n','')
            line = line.replace('cosmo ','')
            xs.Xset.cosmo = ' '.join(line.split(','))
            line = model_file.readline()
        if 'statistic' in line:
            line = model_file.readline()
            continue
        if 'method' in line:
            line = model_file.readline()
            continue
        if 'delta' in line:
            line = model_file.readline()
            continue
        if 'systematic' in line:
            line = model_file.readline()
            continue
        if 'abund' in line:
            line = line.replace('\n','')
            line = line.replace('abund ','')
            xs.Xset.abund = ' '.join(line.split(','))
            line = model_file.readline()
        if 'xsect' in line:
            line = line.replace('\n','')
            line = line.replace('xsect ','')
            xs.Xset.xsect = ' '.join(line.split(','))
            line = model_file.readline()
        
    line = line.replace('\n','')
    line = line.replace('model','')
    
    if not pars_only:
        m = xs.Model(line)
        nb_para = m.nParameters
        #On assigne les parametres
        i=1
        line = model_file.readline()
        while (line and i<=nb_para):
            line = line.replace('\n','')
            param_list = line.split()
            param_list = list(map(float,param_list))
            m(i).values =  param_list
            line=model_file.readline()
            i+=1
        
        return m
    
    else:
        mod_name=line
        Pars=[]
        Frozen=[]
        i=1
        line = model_file.readline()
        while (line):
            line = line.replace('\n','')
            param_list = line.split()
            param_list = list(map(float,param_list))
            Pars.append(param_list[0])
            if param_list[1]<0:
                Frozen.append(i)
            line=model_file.readline()
            i+=1
        
        return mod_name, Pars, Frozen
    
def shakefit(dchi,model):
    nopar = model.nParameters
    xs.Fit.nIterations=100
    pattern = re.compile(r'F........')
    pattern2 = re.compile(r'.T.......')
    
    for j in range(nopar):
        if (model(j+1).frozen):
            continue 
        doerror=1
        delchi=dchi
        while (doerror == 1) :
            error_command = 'stopat 100 0.01 max 20.0 %.2f %d' % (delchi,j+1) #changed 20 to 50 (Edo)
            #print error_command
            xs.Fit.error(error_command)
            error_out = model(j+1).error[2]
            if (pattern.match(error_out)):
                doerror=0
            else:
                xs.Fit.perform()
            
            if (pattern2.match(error_out)):
                delchi*=2
                
class FitSpectra():
    def __init__(self,
                model_file,
                binning,
                background=False,
                E_min = 0.2,
                E_max = 12.,
                NXB_requirement = 5e-3,
                astro_bkg_model = None, 
                cosmic_bkg_model = None,
                xifu_config = XIFU_Config()):
        
        self.model_file = model_file
        self.binning = binning
        self.E_min = E_min
        self.E_max = E_max
        self.background = background
        self.NXB_requirement = NXB_requirement
        self.astro_bkg_model = astro_bkg_model
        self.cosmic_bkg_model = cosmic_bkg_model
        self.xifu_pixel_surface_cm2 = (xifu_config.pixel_size_m*100)**2
        self.xifu_pixel_size_arcmin = (xifu_config.pixsize_degree).to(units.arcmin).value
        
    def fit_spectrum(self,
                    pha_file,
                    arf_file,
                    pix_in_region,
                    log_fitting = True,
                    chatter = 10,
                    do_shakefit = True):
              
        xs.Xset.chatter = chatter
        xs.AllChains.clear()
        xs.AllData.clear()
        xs.AllModels.clear()

        #xs.Xset.restore(self.model_file)
        #m = xs.AllModels(1)
        m = load_model(self.model_file)

        logfile = pha_file.replace('.pha','.log')
        if log_fitting :
            xs.Xset.logChatter = 10
            xs.Xset.openLog(logfile)

        s = xs.Spectrum(pha_file)
        s.response.arf = arf_file

        # Option to run the background
        if self.background: 

            ## NXB 

            #Value in cts/s/keV for a given pixel times the number of pixels
            NXB_level = self.NXB_requirement * self.xifu_pixel_surface_cm2 * pix_in_region
            s.multiresponse[1]=s.multiresponse[0].rmf
            m2=xs.Model("pow", modName="nxb", sourceNum=2, setPars={1:0,2:NXB_level})
            m2(1).frozen=True #constant

            ## AXB
            if self.astro_bkg_model is not None:
                s.multiresponse[2]=s.multiresponse[0].rmf
                s.multiresponse[2].arf=s.multiresponse[0].arf

                axb_model, axb_pars, axb_frozen = load_model(self.astro_bkg_model, pars_only=True)
                m3=xs.Model(axb_model, modName="axb", sourceNum=3, setPars=axb_pars)

                # Freeze parameters
                for j in axb_frozen: 
                    m3(j).frozen=True 

                # Multiply the norm by the angular size of the region
                region_size_arcmin = self.xifu_pixel_size_arcmin**2 * pix_in_region
                m3.setPars({4:m3(4).values[0]*region_size_arcmin})
                m3.setPars({9:m3(9).values[0]*region_size_arcmin})

            ## CXB
            if self.cosmic_bkg_model is not None:
                s.multiresponse[3]=s.multiresponse[0].rmf
                s.multiresponse[3].arf=s.multiresponse[0].arf

                cxb_model, cxb_pars, cxb_frozen=load_model(self.cosmic_bkg_model, pars_only=True)
                m4=xs.Model(cxb_model, modName="cxb", sourceNum=4, setPars=cxb_pars)

                # Freeze parameters
                for j in cxb_frozen: 
                    m4(j).frozen=True

                # Multiply the norm by the angular size of the region
                region_size_arcmin = self.xifu_pixel_size_arcmin**2 * pix_in_region
                m4.setPars({3:m4(3).values[0]*region_size_arcmin})

        # Fitting
        xs.AllData.ignore('0.0-%.1f' % self.E_min)
        xs.AllData.ignore('%.1f-**' % self.E_max)

        xs.AllData.ignore('bad')
        xs.Fit.statMethod = "cstat"

        xs.Fit.nIterations = 1000   
        xs.Fit.query = 'yes'

        xs.Fit.perform()
        xs.Fit.perform()
        xs.Fit.perform()

        # Call shakefit to avoid local minima in case error computation has been requested
        if do_shakefit:
            shakefit(1,m)

        # Return chi2 if this is a nested fit
        chi2 = xs.Fit.statistic

        # Get fit results
        m.show()

        pars_dict = {}
        for k in range(1,1+m.nParameters):
            if k == 1:
                pars_dict[m(k).name] = m(k).values[0]
            elif k == 6 :
                pars_dict[m(k).name] = m(k).values[0] / pix_in_region
                pars_dict[m(k).name + '_lo'] = m(k).error[0] / pix_in_region
                pars_dict[m(k).name + '_hi'] = m(k).error[1] / pix_in_region
            else :
                pars_dict[m(k).name] = m(k).values[0]
                pars_dict[m(k).name + '_lo'] = m(k).error[0]
                pars_dict[m(k).name + '_hi'] = m(k).error[1]
        pars_dict['chi2'] = chi2
        
        if self.background: 
            pars_dict['nxb_norm'] = m2(2).values[0] / pix_in_region / self.xifu_pixel_surface_m
            
            if self.astro_bkg_model is not None:
                pars_dict['axb_norm1'] = m3(4).values[0] / pix_in_region / self.xifu_pixel_size_arcmin**2
                pars_dict['axb_norm2'] = m3(9).values[0] / pix_in_region / self.xifu_pixel_size_arcmin**2
                
            if self.cosmic_bkg_model is not None:
                pars_dict['cxb_norm'] = m4(3).values[0] / pix_in_region / self.xifu_pixel_size_arcmin**2
        
        # End logging
        if log_fitting :
            xs.Xset.closeLog()  
        return pars_dict
    
    def fit_all_spectra(self,
                        spectra_path,
                        numproc = 1,
                        log_fitting = False,
                        chatter = 0,
                        save_fit_results = False,
                        do_shakefit = True):
        
        # Test value of 32
        n_bins = self.binning.nb_bins
        args_list = zip([spectra_path + 'spec_{}.pha'.format(k) for k in range(n_bins)],
                        [spectra_path + 'spec_{}.arf'.format(k) for k in range(n_bins)],
                        [len(self.binning.binning_dict[k][0]) for k in range(n_bins)])
        
        results = Parallel(n_jobs=numproc, 
                           backend="loky")(
                                delayed(self.fit_spectrum)(pha_file, 
                                                           arf_file,
                                                           nb_pix_in_bin,
                                                           log_fitting = log_fitting,
                                                           chatter = chatter,
                                                           do_shakefit = do_shakefit) for pha_file, arf_file, nb_pix_in_bin in tqdm_notebook(args_list, 
                                                                                                                               total = n_bins, 
                                                                                                                               desc = 'Fitting spectra'
                                                                                                                               )
                                )
        self.best_fit_values = results

        rows = [
                    {k: (v[0] if isinstance(v, (list, tuple)) else v) for k, v in d.items()}
                    for d in results
                ]

        df_results = pd.DataFrame(rows)


        if save_fit_results :
            #np.save(spectra_path + 'fit_all_spectra_res.npy', np.array(results))
            df.to_csv(spectra_path + 'fit_all_spectra_res.csv')

        return df_results


    def make_bestfit_maps(self,
                          maps_path,
                          cluster_redshift = None,
                        save_maps = False):
        
        best_fit_norm = np.zeros(self.binning.shape)
        best_fit_kT = np.zeros(self.binning.shape)
        best_fit_Z = np.zeros(self.binning.shape)
        best_fit_z = np.zeros(self.binning.shape)
        best_fit_v = np.zeros(self.binning.shape)
        best_fit_broad = np.zeros(self.binning.shape)
        
        norm_best_fits = np.array([bestfit['norm'][0] for bestfit in self.best_fit_values])
        kT_best_fits = np.array([bestfit['kT'][0] for bestfit in self.best_fit_values])
        Z_best_fits = np.array([bestfit['Abundanc'][0] for bestfit in self.best_fit_values])
        z_best_fits = np.array([bestfit['Redshift'][0] for bestfit in self.best_fit_values])
        broad_best_fits = np.array([bestfit['Velocity'][0] for bestfit in self.best_fit_values])
        
        best_fit_norm[self.binning.X_pixels,
                      self.binning.Y_pixels] = norm_best_fits[self.binning.bin_num_pix]
        best_fit_kT[self.binning.X_pixels,
                    self.binning.Y_pixels] = kT_best_fits[self.binning.bin_num_pix]
        best_fit_Z[self.binning.X_pixels,
                   self.binning.Y_pixels] = Z_best_fits[self.binning.bin_num_pix]
        best_fit_z[self.binning.X_pixels,
                   self.binning.Y_pixels] = z_best_fits[self.binning.bin_num_pix]
        best_fit_broad[self.binning.X_pixels,
                       self.binning.Y_pixels] = broad_best_fits[self.binning.bin_num_pix]
        
        if cluster_redshift == None :
            raise Exception("Need to provide redshift of cluster")
        else :
            num = (1+best_fit_z)**2 - (1+cluster_redshift)**2
            denom = (1+best_fit_z)**2 + (1+cluster_redshift)**2
            best_fit_v = const.c.to(units.km/units.s).value * num/denom
        
        nb_pixels_per_bin = np.array(
            [len(self.binning.binning_dict[k][0]
                                    ) for k in range(self.binning.nb_bins)])

        best_fit_norm[self.binning.X_pixels,
                      self.binning.Y_pixels] /= nb_pixels_per_bin[self.binning.bin_num_pix]

        if save_maps:
            np.savez_compressed(maps_path + 'output_maps.npz', 
                                output_norm = best_fit_norm,
                                output_kT = best_fit_kT,
                                output_Z = best_fit_Z,
                                output_z = best_fit_z,
                                output_v = best_fit_v,
                                output_broad = best_fit_broad)
            
        return best_fit_norm,best_fit_kT,best_fit_Z,best_fit_z,best_fit_v,best_fit_broad
        