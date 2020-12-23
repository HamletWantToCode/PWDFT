import numpy as np 
from .potential import random_periodic_potential

def hamilton_builder(k: float, vq: np.array, matrix_size: tuple):
    """
    At each k point, we generate a single particle Hamiltonian according to:
    H(k)_{m,n} = \frac{(k+G)^2}{2}\delta_{m,n} + V(G_m-G_n)
    """
    Hkq_cache = np.zeros(matrix_size)

    for i in range(1, len(vq)):
        np.fill_diagonal(Hkq_cache[i:, :-i], vq[i])

    total_num_G = matrix_size[0]
    Gs = np.fft.fftshift(np.fft.fftfreq(total_num_G, d=1.0/total_num_G))
    _kinetic_energy = -0.5*(k + 2*np.pi*Gs)**2
    np.fill_diagonal(Hkq_cache, _kinetic_energy)

    return Hkq_cache

def get_spectrum_and_density_matrix_at_k(Hkq: np.array, chem_potential: float):
    """
    We will diagonalize the Hamiltonian:
    \Lambda, U = eigen(Hk)
    """
    spectrum_k, Uk = np.linalg.eigh(Hkq)
    # Fermi statistics
    occupation_num = sum(spectrum_k <= chem_potential)
    rho_kq = Uk @ Uk.conj().T
    return spectrum_k, rho_kq

def get_kinetic_energy_density_at_k(Hkq: np.array, rho_kq: np.array):
    return np.einsum("i,ii->", np.diag(Hkq), rho_kq)

def get_electron_density_at_k(rho_kq: np.array):
    num_of_positive_G = rho_kq.shape[0]
    density_kq_cache = np.zeros(num_of_positive_G)

    for i in range(num_of_positive_G):
        density_kq_cache[i] = np.trace(rho_kq, k=-i))
    
    return density_kq_cache




