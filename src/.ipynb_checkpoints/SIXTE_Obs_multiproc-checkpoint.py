import time
import numpy as np
import sys
sys.path.append('/xifu/home/mola/turb_simsoft/xifusims')
from run_xifupipeline import run_xifupipeline_parts
### In this code I want to execute the function that creates a photon list
### from a slice of input table. I also want to time the operation to evaluate
### its speed

t0 = time.time()

### Define variables and paths

# INSTRUMENT DIRECTORY 
instdir='/xifu/usr/share/sixte/instruments/athena-xifu_2024_11/baseline'

# XML STANDARD FILE
std_xmlfile = instdir+'/xifu_nofilt_infoc.xml'

# XML INSTRUMENT FILE
xmlfile = instdir+'/xifu_detector_adv_20240326.xml'

#path
outdir_simput = './'

#exposure 
exposure = 125e3

#cluster redshift
cluster_redshift = 0.1

#in the original sims we use pointer type =3 
pointer_type = 3

#number of cores used in the regular version of this code
numproc = 32
i = 0

run_xifupipeline_parts(i*numproc, 
                                numproc, 
                                outdir_simput, 
                                simputfile_prefix="ph_list_me", 
                                eventfile_prefix="events", 
                                exposure=exposure,
                                xmlfile=xmlfile,
                                std_xmlfile=std_xmlfile,
                                nb_parts=numproc)
        
t_exec = time.time() - t0

print('Execution time : ',t_exec)