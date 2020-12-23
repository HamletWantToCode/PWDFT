import numpy as np

def to_x(fG, Npoints):
    fx = np.fft.irfft(fG, Npoints) * Npoints
    return fx 

# when V(G) are real, will \rho(G) be real?
def to_G(fx):
    return np.fft.rfft(fx) / len(fx)