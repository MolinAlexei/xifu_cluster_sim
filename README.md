# xifu_cluster_sim
Simulation of observation of galaxy clusters with X-IFU

This is the updated version of the code used in the following papers :

- [Toward mapping turbulence in the intra-cluster medium III. Constraints on the turbulent power spectrum with Athena/X-IFU](https://www.aanda.org/articles/aa/abs/2024/06/aa48937-23/aa48937-23.html)

- [Toward mapping turbulence in the intracluster medium IV. Using NewAthena/X-IFU and simulation-based inference to constrain turbulence](https://www.aanda.org/articles/aa/full_html/2025/10/aa55585-25/aa55585-25.html)


This code can : 
- Simulate a toy-model galaxy cluster on a discret grid (with its density, norm, temperature, abundance, velocity)
- Simulate a random realization of a velocity field drawn from a known power spectrum
- Create a photon list from the list of properties of the cluster (this can be extended to any simulation output). The emission is assumed to be a bapec model, with a single abundance.
- Launch several SIXTE processes in parallel to simulate an X-IFU observation of the cluster.
- Manage the SIXTE outputs to make a single event list, count map, make a Vornoi binning of the map and create spectra for each bin.
- Do the spectral fitting of all synthetic spectra in parallel.
- Create output maps (and velocity structure function)

Please find the documentation [here](https://molinalexei.github.io/xifu_cluster_sim/)