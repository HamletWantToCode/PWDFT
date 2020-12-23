import numpy as np 

__all__ = ["random_periodic_potential"]

"""
Here we generate periodic random potential from this formula:
N = # of Fourier components
G_m = 2\pi m, m\in[-(N-1)//2, N//2]
\hat{V}(G_m) = -\sqrt(2\pi)\sum_{i=1}^n V_i\sigma_i\euler^{-2(\pi\sigma_i m)^2}\cos(2\pi\mu_i m)
V(z) = \sum_m\hat{V}(G_m)\euler^{2\pi j G_m z}, z\in[0, 1)
"""
def random_periodic_potential_inner(
    V0: np.array,
    sigma: np.array,
    mu: np.array,
    nG: int
    ):
    G = np.arange(1, nG+1, 1)
    decay_term = np.exp(-2.0 * ((np.einsum("i,m->im", sigma, np.pi*G))**2))
    oscillate_term = np.cos(2.0*np.pi*(np.einsum("i,m->im", mu, G)))
    amplitude_term = -np.sqrt(2.0*np.pi) * (V0*sigma)[:, None]
    vGs = amplitude_term * decay_term * oscillate_term
    _vG = np.sum(vGs, axis=0)
    vG = np.concatenate((np.zeros(1), _vG))
    return vG

def random_periodic_potential(
    n_component: int,
    nG: int, 
    V0_range: tuple,
    mu_range: tuple,
    sigma_range: tuple,
    seed):

    rng = np.random.RandomState(seed)
    while True:
        V0 = rng.uniform(*V0_range, size=n_component)
        mu = rng.uniform(*mu_range, size=n_component)
        sigma = rng.uniform(*sigma_range, size=n_component)
        vG = random_periodic_potential_inner(V0, sigma, mu, nG)
        yield vG

