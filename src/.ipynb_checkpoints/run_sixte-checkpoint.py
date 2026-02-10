from astropy.io import fits
import numpy as np
from multiprocessing import Process, Pool
import os
import sys
import subprocess

def run_sixte_sim_multiproc(numproc,
                            path = '/xifu/home/mola/xifu_cluster_sim/clusters/obs5_reproduction/',
                            ph_list_prefix="ph_list",
                            evt_file_prefix='event_list',
                            exposure=1e5,
                            std_xmlfile='/data/xifu/usr/SIXTE/share/sixte/instruments/athena/xifudev/xifu_baseline.xml',
                            astro_bkg_simput=None, 
                            cosmic_bkg_simput=None, 
                            background=False):
    '''
    Function to launch one SIXTE process per core 
    using the photon lists generated for a cluster sim.
    Will overwrite existing event files.  
    Matches the syntax used in SIXTE 3.0.5
    Parameters:
        numproc (int): number of processes to run 
        path (str): directory in which to save the resulting ph lists
        ph_list_prefix (str): prefix to use for the simput file names
        evt_file_prefix (str): prefix to use for the event file names
        clobber (bool): standard FITS clobber option
        exposure (float): exposure to use for the photon generation
        std_xmlfile (str): standard XML file to use for the simulations
        background (bool): option to include NXB
        astro_bkg_simput (str): path to simput file with astrophysical background
        cosmic_bkg_simput (str): path to simout file with cosmic AGNs
    '''   

    list_w = []
    for i in range(numproc):
        proc_outdir = path+'/part%d/' % (i)
        simputfile = proc_outdir+ph_list_prefix+'_%d.fits' % (i)
        eventfile = proc_outdir+evt_file_prefix+'_%d.fits' % (i)
        piximpfile = eventfile.replace('event','impact').replace('.evt','.piximp')
        if i==0 and background==True:
            #The background is included only once in one simulation (otherwise nparts*expected background)
            if (astro_bkg_simput is None) and (cosmic_bkg_simput is None):
                list_w.append(subprocess.Popen(['sixtesim',
                                            'ImpactList='+piximpfile,
                                            'XMLFile='+std_xmlfile,
                                            'Background=yes',
                                            'RA=0.0','Dec=0.0',
                                            'Simput='+simputfile, 
                                            'Exposure=%g'%exposure,
                                            'clobber=yes',
                                            'EvtFile=%s' % eventfile]))

            elif (astro_bkg_simput is not None) and (cosmic_bkg_simput is None):
                list_w.append(subprocess.Popen(['sixtesim',
                                            'ImpactList='+piximpfile,
                                            'XMLFile='+std_xmlfile,
                                            'Background=yes',
                                            'RA=0.0','Dec=0.0',
                                            'Simput='+simputfile+','+astro_bkg_simput,
                                            'Exposure=%g'%exposure,
                                            'clobber=yes',
                                            'EvtFile=%s' % eventfile]))

            elif (astro_bkg_simput is None) and (cosmic_bkg_simput is not None):
                list_w.append(subprocess.Popen(['sixtesim',
                                            'ImpactList='+piximpfile,
                                            'XMLFile='+std_xmlfile,
                                            'Background=yes',
                                            'RA=0.0','Dec=0.0',
                                            'Simput='+simputfile+','+cosmic_bkg_simput,
                                            'Exposure=%g'%exposure,
                                            'clobber=yes',
                                            'EvtFile=%s' % eventfile]))
            else:
                list_w.append(subprocess.Popen(['sixtesim',
                                            'ImpactList='+piximpfile,
                                            'XMLFile='+std_xmlfile,
                                            'Background=yes',
                                            'RA=0.0','Dec=0.0',
                                            'Simput='+simputfile+','+astro_bkg_simput+','+cosmic_bkg_simput,
                                            'Exposure=%g'%exposure,
                                            'clobber=yes',
                                            'EvtFile=%s' % eventfile]))
        else:
            print('Command that is being executed : sixtesim',
                'ImpactList='+piximpfile,
                'XMLFile='+std_xmlfile,
                'Background=no',
                'RA=0.0','Dec=0.0',
                'Simput='+simputfile,
                'Exposure=%g'%exposure,
                'clobber=yes',
                'EvtFile=%s' % eventfile)

            list_w.append(subprocess.Popen(['sixtesim',
                                            'ImpactList='+piximpfile,
                                            'XMLFile='+std_xmlfile,
                                            'Background=no',
                                            'RA=0.0','Dec=0.0',
                                            'Simput='+simputfile,
                                            'Exposure=%g'%exposure,
                                            'clobber=yes',
                                            'EvtFile=%s' % eventfile]))
        print("Started %d" % i)
    for w in list_w:
         w.communicate()

def run_sixte_sim_notebook(nparts,
                        path = '/xifu/home/mola/xifu_cluster_sim/clusters/obs5_reproduction/',
                        ph_list_prefix="ph_list",
                        evt_file_prefix='evt_list',
                        exposure=1e5,
                        std_xmlfile='/data/xifu/usr/SIXTE/share/sixte/instruments/athena/xifudev/xifu_baseline.xml',
                        astro_bkg_simput=None, 
                        cosmic_bkg_simput=None, 
                        background=False):
    '''
    Function to launch SIXTE on a notebook (single core) 
    using the photon lists generated for a cluster sim.
    Will overwrite existing event files.  
    Matches the syntax used in SIXTE 3.0.5
    Parameters:
        nparts (int): number of parts of photon lists
        path (str): directory in which to save the resulting ph lists
        ph_list_prefix (str): prefix to use for the simput file names
        evt_file_prefix (str): prefix to use for the event file names
        clobber (bool): standard FITS clobber option
        exposure (float): exposure to use for the photon generation
        std_xmlfile (str): standard XML file to use for the simulations
        background (bool): option to include NXB
        astro_bkg_simput (str): path to simput file with astrophysical background
        cosmic_bkg_simput (str): path to simout file with cosmic AGNs
    '''

    for i in range(nparts):
        proc_outdir = path+'/part%d/' % (i)
        simputfile = proc_outdir+ph_list_prefix+'_%d.fits' % (i)
        eventfile = proc_outdir+evt_file_prefix+'_%d.fits' % (i)
        piximpfile = eventfile.replace('event','impact').replace('.evt','.piximp')
        if i==0 and background==True:
            #The background is included only once in one simulation (otherwise nparts*expected background)
            if (astro_bkg_simput is None) and (cosmic_bkg_simput is None):
                subprocess.check_call(['sixtesim',
                                            'ImpactList='+piximpfile,
                                            'XMLFile='+std_xmlfile,
                                            'Background=yes',
                                            'RA=0.0','Dec=0.0',
                                            'Simput='+simputfile, 
                                            'Exposure=%g'%exposure,
                                            'clobber=yes',
                                            'EvtFile=%s' % eventfile])

            elif (astro_bkg_simput is not None) and (cosmic_bkg_simput is None):
                subprocess.check_call(['sixtesim',
                                            'ImpactList='+piximpfile,
                                            'XMLFile='+std_xmlfile,
                                            'Background=yes',
                                            'RA=0.0','Dec=0.0',
                                            'Simput='+simputfile+','+astro_bkg_simput,
                                            'Exposure=%g'%exposure,
                                            'clobber=yes',
                                            'EvtFile=%s' % eventfile])

            elif (astro_bkg_simput is None) and (cosmic_bkg_simput is not None):
                subprocess.check_call(['sixtesim',
                                            'ImpactList='+piximpfile,
                                            'XMLFile='+std_xmlfile,
                                            'Background=yes',
                                            'RA=0.0','Dec=0.0',
                                            'Simput='+simputfile+','+cosmic_bkg_simput,
                                            'Exposure=%g'%exposure,
                                            'clobber=yes',
                                            'EvtFile=%s' % eventfile])
            else:
                subprocess.check_call(['sixtesim',
                                            'ImpactList='+piximpfile,
                                            'XMLFile='+std_xmlfile,
                                            'Background=yes',
                                            'RA=0.0','Dec=0.0',
                                            'Simput='+simputfile+','+astro_bkg_simput+','+cosmic_bkg_simput,
                                            'Exposure=%g'%exposure,
                                            'clobber=yes',
                                            'EvtFile=%s' % eventfile])
        else:

            subprocess.check_call(['sixtesim',
                                            'ImpactList='+piximpfile,
                                            'XMLFile='+std_xmlfile,
                                            'Background=no',
                                            'RA=0.0','Dec=0.0',
                                            'Simput='+simputfile,
                                            'Exposure=%g'%exposure,
                                            'clobber=yes',
                                            'EvtFile=%s' % eventfile])
    