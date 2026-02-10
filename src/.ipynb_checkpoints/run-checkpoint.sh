        #!/bin/sh 

        #SBATCH --job-name=turbu-pall -N1  -c32                                                                               
        #SBATCH --output=test.log                                                                           
        #SBATCH --partition=xifu

        #mendatory options with turbulent_sim.py : path_bapec_model  path_nlapec_model  OUTPUT_DIR  turbu_model_dir  output_subfolder_dir                         

        export PYTHONPATH="${PYTHONPATH}:/xifu/home/mola/xifu_cluster_sim/"
        export HEADASNOQUERY=
        export HEADASPROMPT=/dev/null

        dir=/xifu/home/mola/xifu_cluster_sim/clusters/obs5_reproduction/
        X=([0])
Y=([0])
pointing_name=(["central_pointing"])
runs=${#Y[@]}

for ((i==0; i<runs; i++)); do

echo ""
echo "Running turbulent_sim for pointing X"${X[i]}" Y"${Y[i]} "named" ${pointing_name[i]}
echo ""



python3 /xifu/home/mola/xifu_cluster_sim/src/launch_pointing_nuwa.py ${dir} ${X[i]} ${Y[i]} ${pointing_name[i]} --velocity_file '/xifu/home/mola/xifu_cluster_sim/src/turbulent_speed_obs5.npy'


done
