from pwdft import to_x, to_G
import numpy as np 
import unittest
import logging

logging.basicConfig(level=logging.WARN)

class TestUtils(unittest.TestCase):
    def test_transform(self):
        Aq = np.random.randn(10) + 1j*np.concatenate((np.zeros(1), np.random.randn(9)))
        Ax = to_x(Aq, 100)
        Aq1 = to_G(Ax)

        logging.debug("Aq={}, Aq1={}".format(Aq, Aq1))
        self.assertTrue(np.allclose(Aq, Aq1[:10]))