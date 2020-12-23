import numpy as np 
from pwdft.physics import *
import unittest

k = 0.0
vq = np.random.randn(5) + 1j*np.random.randn(5)
matrix_size = (10, 10)
Hkq = hamilton_builder(k, vq, matrix_size)

class TestPhysics(unittest.TestCase):
    def test_hamilton(self):
        """
        1. Hermitian
        2. banded
        """
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

    def test_density_matrix(self):
        """
        1. trace(\rho) = occ
        2. \rho^2 = \rho
        3. \rho^{\dagger} = \rho
        """
        mu = -20.0
        spectrum, rho = get_spectrum_and_density_matrix_at_k(Hkq, mu)

        self.assertTrue(spectrum.dtype == np.float64)
        self.assertTrue(np.allclose(np.real(np.trace(rho)), np.sum(spectrum <= mu)))
        self.assertTrue(np.allclose(rho @ rho, rho))
        self.assertTrue(np.allclose(rho.conj(), rho.T))




