import numpy as np 
from pwdft.physics import *
from pwdft.dft import run_dft
import unittest
import logging

logging.basicConfig(level=logging.WARN)

class TestPhysics(unittest.TestCase):
    def test_basic_property(self):
        """
        Hamiltonian should be:
        1. Hermitian
        2. banded

        Density matrix should be:
        1. trace(\rho) = occ
        2. \rho^2 = \rho
        3. \rho^{\dagger} = \rho
        """
        # setups
        k = 0.0
        vq = np.random.randn(5) + 1j*np.random.randn(5)
        matrix_size = (10, 10)
        Hkq = hamilton_builder(k, matrix_size, vq)

        ## Test Hamiltonian
        self.assertTrue(Hkq.dtype == vq.dtype)

        flag = True
        for i in range(1, 5, 1):
            if np.any(np.diag(Hkq, k=-i) != vq[i]):
                flag = False
        
        self.assertTrue(flag)

        flag = True
        for i in range(5, 10, 1):
            if np.any(np.diag(Hkq, k=-i) != 0.0):
                flag = False
        
        self.assertTrue(flag)

        ## Test density matrix
        mu = -20.0
        spectrum, rho = get_spectrum_and_density_matrix_at_k(Hkq, mu)

        self.assertTrue(spectrum.dtype == np.float64)
        self.assertTrue(np.allclose(np.real(np.trace(rho)), np.sum(spectrum <= mu)))
        self.assertTrue(np.allclose(rho @ rho, rho))
        self.assertTrue(np.allclose(rho.conj(), rho.T))
        
    def test_TF_model(self):
        """
        Here we measure two relations:
        1. Thomas-Fermi functional: T[\rho_0] = \frac{\pi^2}{6}\rho_0^3, where \rho_0 is the electron density per cell of free electron system.
        2. \pi\rho_0 = \sqrt(2\mu), where \mu is the chemical potential.
        """
        mu = 20.0
        Nk = 100
        matrix_size = (20, 20)
        kpts = np.linspace(-np.pi, np.pi, Nk)
        
        e_k, rho0cell_q = run_dft(kpts, None, mu, matrix_size)
        rho0cell = rho0cell_q[0]

        logging.debug("\pi*\rho0={}, \sqrt(2\mu)={}, \varepsilon_k={}".format(np.pi*rho0cell, np.sqrt(2*mu), e_k))

        self.assertTrue(np.allclose(e_k, (np.pi**2/6)*rho0cell**3, rtol=1e-2))
        self.assertTrue(np.allclose(np.sqrt(2*mu)/rho0cell, np.pi, rtol=0.1))

    def test_near_free(self):
        """
        Near free electrons experience a periodic potential has the following property:
        1. \Delta E_{21}\approx 2|\hat{V}(G=1)|
        """

        def calc_bandgap_at_K(V0, matrix_size):
            K = np.pi 
            mu = 0.0 # \mu is not important here, we only care about band gap
            Vq = V0 * np.array([-0.5, -0.25])
            HKq = hamilton_builder(K, matrix_size, Vq)
            spectrum, _ = get_spectrum_and_density_matrix_at_k(HKq, mu)
            bandgap = spectrum[1] - spectrum[0]
            return bandgap
        
        V0s = np.arange(0.0, 5.0, 0.1)
        matrix_size = (50, 50)
        bandgaps = []
        for V0 in V0s:
            gap = calc_bandgap_at_K(V0, matrix_size)
            bandgaps.append(gap)
        bandgaps = np.asarray(bandgaps)
        coeffs = np.polyfit(0.25*V0s, bandgaps, 1)
        logging.debug("k={}, b={}".format(coeffs[0], coeffs[1]))
        
        self.assertTrue(np.allclose(coeffs[0], 2.0, rtol=1e-2))
        self.assertTrue(np.allclose(coeffs[1], 0.0, atol=1e-2))






