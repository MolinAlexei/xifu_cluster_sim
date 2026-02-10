import numpy as np
from astropy.cosmology import LambdaCDM
from astropy import units as units
import astropy.constants as const
from astropy.io import fits

import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True) ## NECESSARY FOR MY COMPUTATION OF THE NORM

from scipy.linalg import solve

import subprocess

from joblib import Parallel, delayed
import joblib
from tqdm.notebook import tqdm as tqdm_notebook

def quadratic_from_bin_averages(E_edges, V_row, boundary='natural', reg_eps=1e-12):
    """
    Build piecewise quadratic per bin with exact bin averages and C1 continuity.
    boundary: 'zero_slope', 'natural', or 'extrapolate'.
      - 'zero_slope': enforce derivative = 0 at both ends
      - 'natural'   : enforce second derivative = 0 at both ends (a_0 = a_last = 0)
      - 'extrapolate': set right-end derivative from trend of last bins (left derivative = 0)
    """
    E_edges = np.asarray(E_edges, dtype=float)
    V_row = np.asarray(V_row, dtype=float)
    n_bins = len(V_row)
    n_unknowns = 3 * n_bins

    # prepare system
    A = np.zeros((n_unknowns, n_unknowns), dtype=float)
    rhs = np.zeros(n_unknowns, dtype=float)
    row_idx = 0

    def avg_basis(Δ):
        return np.array([Δ**2 / 3.0, Δ / 2.0, 1.0], dtype=float)

    # 1) bin-average constraints
    for i in range(n_bins):
        Ei = E_edges[i]; Ej = E_edges[i+1]; Δ = Ej - Ei
        A[row_idx, 3*i:3*i+3] = avg_basis(Δ)
        rhs[row_idx] = V_row[i]
        row_idx += 1

    # 2) continuity constraints (value + slope) between bins
    for i in range(n_bins - 1):
        Ei = E_edges[i]; Ej = E_edges[i+1]; Δ = Ej - Ei
        # value continuity: a_i*Δ^2 + b_i*Δ + c_i - c_{i+1} = 0
        A[row_idx, 3*i:3*i+3] = [Δ**2, Δ, 1.0]
        A[row_idx, 3*(i+1) + 2] = -1.0
        rhs[row_idx] = 0.0
        row_idx += 1
        # slope continuity: 2*a_i*Δ + b_i - b_{i+1} = 0
        A[row_idx, 3*i:3*i+3] = [2.0*Δ, 1.0, 0.0]
        A[row_idx, 3*(i+1) + 1] = -1.0
        rhs[row_idx] = 0.0
        row_idx += 1

    # 3) boundary conditions
    if boundary == 'zero_slope':
        # left derivative b_0 = 0
        A[row_idx, 1] = 1.0
        rhs[row_idx] = 0.0
        row_idx += 1
        # right derivative at end: 2*a_last*Δ_last + b_last = 0
        Δ_last = E_edges[-1] - E_edges[-2]
        A[row_idx, -3:] = [2.0*Δ_last, 1.0, 0.0]  # FIXED
        rhs[row_idx] = 0.0
        row_idx += 1

    elif boundary == 'natural':
        # natural: second derivative = 0 at left and right -> a_0 = 0, a_last = 0
        A[row_idx, 0] = 1.0
        rhs[row_idx] = 0.0
        row_idx += 1
        A[row_idx, -3] = 1.0
        rhs[row_idx] = 0.0
        row_idx += 1

    elif boundary == 'extrapolate':
        # left derivative = 0
        A[row_idx, 1] = 1.0
        rhs[row_idx] = 0.0
        row_idx += 1
        # estimate slope at right
        centers = 0.5*(E_edges[:-1] + E_edges[1:])
        if len(centers) >= 2:
            slope_est = (V_row[-1] - V_row[-2]) / (centers[-1] - centers[-2])
        else:
            slope_est = 0.0
        Δ_last = E_edges[-1] - E_edges[-2]
        A[row_idx, -3:] = [2.0*Δ_last, 1.0, 0.0]  # FIXED
        rhs[row_idx] = slope_est
        row_idx += 1

    else:
        raise ValueError("boundary must be 'zero_slope', 'natural', or 'extrapolate'")

    assert row_idx == n_unknowns, f"constructed {row_idx} rows but need {n_unknowns}"

    # Solve robustly
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        ATA = A.T @ A
        ATb = A.T @ rhs
        ATA_reg = ATA + reg_eps * np.eye(ATA.shape[0])
        try:
            sol = np.linalg.solve(ATA_reg, ATb)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    coeffs = sol.reshape(n_bins, 3)
    return coeffs

def eval_quadratic_piecewise(E_edges, coeffs, E_eval):
    E_edges = np.asarray(E_edges, dtype=float)
    E_eval = np.asarray(E_eval, dtype=float)
    vals = np.zeros_like(E_eval, dtype=float)
    for i in range(len(coeffs)):
        a,b,c = coeffs[i]
        left = E_edges[i]; right = E_edges[i+1]
        mask = (E_eval >= left) & (E_eval <= right)
        if not np.any(mask):
            continue
        x = E_eval[mask] - left
        vals[mask] = a*x**2 + b*x + c
    vals[(E_eval < E_edges[0]) | (E_eval > E_edges[-1])] = np.nan
    return vals



class MakeSpectra:
    def __init__(self,
                binning):
        self.binning = binning
        
    def load_vignetting(self,
                       vign_file_path = '/xifu/usr/share/sixte/instruments/athena-xifu_2024_11/baseline/instdata/athena_vig_13rows_20240326.fits',
                       arf_file_path = '/xifu/home/mola/XIFU_Sims_Turbulence_NewConfig/ARF.arf'):
        
        # Load vignetting file and assign arrays
        hdu_vign=fits.open(vign_file_path)
        vign=hdu_vign[1].data
        self.vign_E_centers = (vign['ENERG_LO'] + vign['ENERG_HI'])[0]/2
        self.vign_E_widths = (vign['ENERG_HI'] - vign['ENERG_LO'])[0]
        self.vign_E_bins = np.hstack((vign['ENERG_LO'][0], vign['ENERG_HI'][0][-1]))
        self.vignet_integre = np.array(vign['VIGNET'][0,0])
        self.theta = np.array(vign['THETA'][0])*60. #Arcminutes
        
        # Load arf data
        self.arf_file_path = arf_file_path
        hdu_arf = fits.open(arf_file_path)
        arf = hdu_arf[1].data
        self.arf_specresp = np.array(arf['SPECRESP'])
        self.E_arf = np.array(arf['ENERG_LO'] + arf['ENERG_HI'])/2
        
        self.vign_on_arf_bins = np.zeros(
        								(self.theta.shape[0],
                                    	 self.E_arf.shape[0]
                                    	)
        								)

        for i in range(self.theta.shape[0]):
            coeffs = quadratic_from_bin_averages(self.vign_E_bins, 
                                                 self.vignet_integre[i], 
                                                 boundary = 'natural')
            
            self.vign_on_arf_bins[i] = eval_quadratic_piecewise(self.vign_E_bins, 
                                                           coeffs, 
                                                           self.E_arf) 
        
        
    def load_event(self,
                   event_file_path,
                   grades_to_keep = [1]):
        
        hdu_evt = fits.open(event_file_path)
        self.event_file_path = event_file_path
        self.evt_data = hdu_evt[1].data
        self.evt_data = self.evt_data[np.isin(self.evt_data['GRADING'], grades_to_keep)] 

        
        # Angular distance from center in arcminutes
        self.evt_theta = np.degrees(
                                            np.arccos(
                                                np.cos(np.radians(self.evt_data['DEC'])) * np.cos(np.radians(self.evt_data['RA']))
                                                    ) 
                                            ) * 60
        
    def make_arfs_and_spectra(self, 
                              spectra_path,
                              numproc = 1):
          
        
        results = Parallel(n_jobs=numproc, 
                           prefer="threads",
                          batch_size = 'auto')(
                                delayed(
                                    self.make_single_arf_and_spectrum
                                        )(bin_number, 
                                        spectra_path
                                        ) for bin_number in tqdm_notebook(np.arange(self.binning.nb_bins), 
                                                                          total = self.binning.nb_bins, 
                                                                          desc = 'Making arfs and spectra'
                                                                         )
                                )
    
    def make_single_arf_and_spectrum(self, 
                                     bin_number,
                                     spectra_path):
        
        pixels_in_region=self.binning.binning_dict[bin_number][0]
        interpolated_vignet = np.zeros_like(self.E_arf)
        counts_tot = 0
        for i in range(0, len(pixels_in_region)):
            
            # Real pixel number
            pix_num=pixels_in_region[i] 
            
            
            #Compute theta
            theta_pix= np.mean(self.evt_theta[self.evt_data['PIXID'] == pix_num])

            # Weight with counts
            counts_in_band=np.sum(self.evt_data['PIXID'] == pix_num)
            counts_tot += counts_in_band

            # Broadcast interpolation on axis of energies
            func_to_vmap = lambda vignet_pt : jnp.interp(theta_pix, self.theta, vignet_pt) 
            vmapped_interp = jax.vmap(func_to_vmap)
            interpolated_vignet += vmapped_interp(self.vign_on_arf_bins.T) * counts_in_band 

        interpolated_vignet /= counts_tot
        
        # Load and save modified ARF
        hdu_arf = fits.open(self.arf_file_path)
        hdu_arf[1].data['SPECRESP'] = self.arf_specresp * np.array(interpolated_vignet)
        hdu_arf.writeto(spectra_path + 'spec_{}.arf'.format(bin_number), overwrite=True)
        
        # Load and save modified event
        hdu_evt = fits.open(self.event_file_path)
        hdu_evt['EVENTS'].header['ANCRFILE']= spectra_path + '/spec_{}.arf'.format(bin_number)
        hdu_evt['EVENTS'].header['RESPFILE']= '/xifu/home/mola/XIFU_Sims_Turbulence_NewConfig/RMF.rmf'
        
        hdu_evt[1].data = self.evt_data[np.isin(self.evt_data['PIXID'],
                                                     pixels_in_region)]
        
        evt_filename = spectra_path + '/spec_{}.evt'.format(bin_number)
        hdu_evt.writeto(evt_filename, overwrite=True)
        
        #Make spectrum using makespec
        pha_filename = evt_filename.replace('.evt','.pha')
        arguments=["makespec",
                   "EvtFile=%s" % evt_filename,
                   "Spectrum=%s" % pha_filename,
                   "clobber=T"]
        print("Calling:",arguments)
        subprocess.check_call(arguments)
        
        return None
        