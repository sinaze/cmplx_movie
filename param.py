import numpy as np
from scipy.special import iv

# water
eta = 1e-3        # Pa * s
eta_p = 3e-3      # Pa * s
rho = 1e3         # kg / m**3
c_bulk = 1.5e3    # m / s
K = rho * c_bulk**2
# g = 9.81          # m / s**2
g = 0

# membrane
R = 1e-3
sigma_2d = 10e-3  # N / m
# sigma_2d = 0
K_2d = 10e-3      # N / m
# K_2d = 0

# rho_2d = 1e-6     # kg / m**3
# eta_2d = 1e-9     # Pa * s * m
# eta_2d_p = 0
# eta_2d_t = eta_2d
# kappa_2d = 3e-19
rho_2d = 0
eta_2d = 0
eta_2d_p = 0
eta_2d_t = 0
kappa_2d = 0
# w = 1   # dummy


def alpha_sq(w):
    return (-1j * w) * rho / eta


def lambda_t(k, w):
    return np.sqrt(k**2 + alpha_sq(w))


def f(k, w):
    a = 4j * w**2 * eta**2 * lambda_t(k, w) * k**3
    b1 = 1j * w**2 * 2*eta * k**3 / R \
        + 1j * (K_2d * k**2 + rho_2d * w**2) * k**2 * 2*eta * w / R
    b2 = k**2 * (K_2d * k**2 + rho_2d * w**2)\
        * (sigma_2d * k**2 - rho_2d * w**2)
    c1 = w * k * 2*eta * lambda_t(k, w) * (w * 2*eta * k**2 - 1)
    c2 = k * lambda_t(k, w) * (sigma_2d * k**2 - w**2 * rho_2d) \
        * (K_2d * k**2 + rho_2d * w**2)
    c3 = 1j * k * w * 2*eta * lambda_t(k, w)\
        * (K_2d * k**2 + rho_2d * w**2) / R
    d1 = -w * k**2 * (K_2d * k**2 - w**2 * rho_2d + 1j * w * 2*eta / R) / R
    d2 = 2*eta * w * k**3\
        * (2*eta * w / R + 1j * (K_2d * k**2 - w**2 * rho_2d))
    z1 = iv(0, k*R) * iv(0, lambda_t(k, w)*R) * a
    z2 = iv(0, k*R) * iv(1, lambda_t(k, w)*R) * (b1 - b2)
    z3 = iv(1, k*R) * iv(0, lambda_t(k, w)*R) * (c1 - c2 - c3)
    z4 = iv(1, k*R) * iv(1, lambda_t(k, w)*R) * (d1 - d2)
    return z1 + z2 + z3 + z4
