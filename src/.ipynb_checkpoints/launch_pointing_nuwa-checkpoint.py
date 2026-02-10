from mosaic import Mosaic
import argparse

parser = argparse.ArgumentParser(description='Runs a complete simulation of X-IFU observation of a cluster')

parser.add_argument("sim_path", 
	type= str,
	help = "Path to the folder of the cluster sim")
parser.add_argument("--x_offsets",
	nargs = '+',
	type = int,
	help = "Pointing offsets along x direction")
parser.add_argument("--y_offsets",
	nargs = '+',
	type = int,
	help = "Pointing offsets along y direction")
parser.add_argument("--pointing_names",
	nargs = '+',
	type = str,
	help = "Names of the pointings")
parser.add_argument("--mosaic_shape", 
	type = tuple, 
	help = "Shape of the full mosaic", 
	default = (232,232))
parser.add_argument("--velocity_file", 
	type = str, 
	help = "Path to the velocity file to load, if any", 
	default = None)
parser.add_argument("--interp_files_path", 
	type = str, 
	help = "Path to the interpolated files of the spectra", 
	default = '/xifu/home/mola/XIFU_Sims_Turbulence_NewConfig/Observation5/files/')

args = parser.parse_args()

print(args.x_offsets)

mosaic = Mosaic('/xifu/home/mola/xifu_cluster_sim/clusters/obs5_reproduction/',
             	velocity_file = args.velocity_file,
             	binning_file = None,
                mosaic_shape = args.mosaic_shape,
                interp_files_path = args.interp_files_path
                )


mosaic.run_mosaic(args.x_offsets,
	args.y_offsets,
	args.pointing_names)