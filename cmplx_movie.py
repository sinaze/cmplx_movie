#!/usr/bin/env python3
"""Visualize complex function using domain coloring and make a movie
along one real parameter (e.g. dispersion relation vs frequency).
"""
import numpy as np
import argparse
import os
import cplot
from scipy.special import iv
from joblib import Parallel, delayed
from param import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('x_min', help='Re(k) min', type=int)
parser.add_argument('x_max', help='Re(k) max', type=int)
parser.add_argument('y_min', help='Im(k) min', type=int)
parser.add_argument('y_max', help='Im(k) max', type=int)
parser.add_argument('w_min', help='log(omega) min', type=int)
parser.add_argument('w_max', help='log(omega) max', type=int)
parser.add_argument('-num', '--number', help='no. of points, default: 1000',
                    type=int, default=1000)
args = parser.parse_args()


def f(k):
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


W = np.logspace(args.w_min, args.w_max, num=args.number)
Parallel(n_jobs=-1, verbose=10, batch_size=1)(
    delayed(cplot.complex_save)(f,
                                (args.x_min, args.x_max),
                                (args.y_min, args.y_max),
                                'img' + str(i) + '.png')
    for i, w0 in enumerate(W))
os.system("ffmpeg -framerate 30 -i img%01d.png\
          -c:v libx264 -pix_fmt yuv420p out.mp4")
os.system("rm *.png")
