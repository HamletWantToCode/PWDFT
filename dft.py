import numpy as np 
from .physics import *

def run_dft(kpoints: np.array,
    vq: np.array,
    chem_potential: float,
    matrix_size: tuple):

    Nk = len(kpoints)
    kinetic_energy_density_per_cell = 0.0
    electron_density_per_cell = np.zeros(matrix_size[0])

    for k in kpoints:
        Hkq = hamilton_builder(k, vq, matrix_size)
        spectrum, density_matrix_kq = get_spectrum_and_density_matrix_at_k(Hkq, chem_potential)
        epsilon_k = get_kinetic_energy_density_at_k(Hkq, density_matrix_kq)
        density_kq = get_electron_density_at_k(density_matrix_kq)

        kinetic_energy_density_per_cell += epsilon_k / Nk
        electron_density_per_cell += density_kq / Nk
    
    return kinetic_energy_density_per_cell, electron_density_per_cell


