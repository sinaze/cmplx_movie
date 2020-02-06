import numpy as np

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
