import argparse
import numpy as np 
from mpi4py import MPI 
from pwdft import run_dft, random_periodic_potential
import h5py
import time
import json
import logging

logging.basicConfig(level=logging.DEBUG)

def main(args):
    comm = MPI.COMM_WORLD
    n_cores = comm.Get_size()
    total_samples = args.N
    sample_per_core = total_samples // n_cores
    core_id = comm.Get_rank()

    # generate random seed for potential generation
    if core_id == 0:
        overall_seed = np.random.seed(12345)
        overall_random_state = np.random.RandomState(overall_seed)
        seeds_integer = overall_random_state.randint(100, 1000, n_cores)
    else:
        seeds_integer = None
    myseed = comm.scatter(seeds_integer, root=0)
    logging.debug("At process {}, my seed is {}, I'm going to calculate {} of data".format(core_id, myseed, sample_per_core))
    rng = np.random.RandomState(myseed)

    qm_config_file = args.config
    # generate random periodic potential
    with open(qm_config_file, "r") as f:
        qm_config = json.load(f)
    V0_range = qm_config["potential"]["V0_range"]
    mu_range = qm_config["potential"]["mu_range"]
    sigma_range = qm_config["potential"]["sigma_range"]
    nG = qm_config["potential"]["nG"]
    n_components = qm_config["potential"]["n_components"]
    random_pp = random_periodic_potential(n_components, nG, V0_range, mu_range, sigma_range, rng)

    # DFT
    nkpts = qm_config["dft"]["nkpts"]
    chem_potential = qm_config["dft"]["chem_potential"]
    matrix_size = qm_config["dft"]["matrix_size"]
    kpoints = np.linspace(-np.pi, np.pi, nkpts)

    # setup cache
    logging.debug("allocating memory at process %d" %(core_id))
    POTENTIAL_CACHE = np.zeros((sample_per_core, nG+1), dtype=np.float64)
    DENSITY_CACHE = np.zeros((sample_per_core, matrix_size[0]), dtype=np.float64)
    EK_CACHE = np.zeros(sample_per_core, dtype=np.float64)

    for i in range(sample_per_core):
        vq = next(random_pp)
        ek, rho_q = run_dft(kpoints, vq, chem_potential, matrix_size)
        POTENTIAL_CACHE[i, :] = vq
        DENSITY_CACHE[i, :] = rho_q
        EK_CACHE[i] = ek
    
    ALL_DENSITY = None
    ALL_POTENTIAL = None
    ALL_EK = None
    if core_id == 0:
        logging.debug("Allocating memory for data collection at process %d" %(core_id))
        ALL_DENSITY = np.zeros((total_samples, matrix_size[0]), dtype=np.float64)
        ALL_POTENTIAL = np.zeros((total_samples, nG+1), dtype=np.float64)
        ALL_EK = np.zeros(total_samples, dtype=np.float64)
    comm.Gather(POTENTIAL_CACHE, ALL_POTENTIAL, root=0)
    comm.Gather(DENSITY_CACHE, ALL_DENSITY, root=0)
    comm.Gather(EK_CACHE, ALL_EK, root=0)

    output_path = args.o
    if core_id == 0:
        logging.debug("Writing data file at process %d" %(core_id))
        current_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
        f = h5py.File("%s/%s.h5" %(output_path, current_time), "w")
        f.create_dataset("density", data=ALL_DENSITY)
        f["density"].attrs["chem_potential"] = chem_potential
        f.create_dataset("potential", data=ALL_POTENTIAL)
        f["potential"].attrs["abs_V0_min"] = V0_range[0]
        f["potential"].attrs["abs_V0_max"] = V0_range[1]
        f["potential"].attrs["mu_min"] = mu_range[0]
        f["potential"].attrs["mu_max"] = mu_range[1]
        f["potential"].attrs["sigma_min"] = sigma_range[0]
        f["potential"].attrs["sigma_max"] = sigma_range[1]
        f.create_dataset("kinetic_energy", data=ALL_EK)
        logging.debug("Finish execution!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, required=True, help="Number of data samples to be generated")
    parser.add_argument("--config", required=True, help="Path to DFT configuration file")
    parser.add_argument("-o", required=True, help="Path to save data file")
    args = parser.parse_args()
    main(args)
