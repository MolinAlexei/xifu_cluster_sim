#!/bin/sh 

#SBATCH --job-name=phlist_test -N1  -c32                                                                     
#SBATCH --output=SIXTE_Obs.log                                                                       
#SBATCH --partition=xifu

#mendatory options with turbulent_sim.py : path_bapec_model  path_nlapec_model  OUTPUT_DIR  turbu_model_dir  output_subfolder_dir                         

export PYTHONPATH="${PYTHONPATH}:/xifu/home/mola/turb_simsoft/"
export HEADASNOQUERY=
export HEADASPROMPT=/dev/null


python3 SIXTE_Obs_multiproc.py 

done