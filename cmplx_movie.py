#!/usr/bin/env python3
"""Visualize complex function using domain coloring and make a movie
along one real parameter (e.g. dispersion relation vs frequency).
"""
import numpy as np
import argparse
import os
import cplot
from joblib import Parallel, delayed
from param import f

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

W = np.logspace(args.w_min, args.w_max, num=args.number)
Parallel(n_jobs=8, verbose=10, batch_size='auto')(
    delayed(cplot.complex_save)(f, w0,
                                (args.x_min, args.x_max),
                                (args.y_min, args.y_max),
                                'img' + str(i) + '.png')
    for i, w0 in enumerate(W))
os.system("ffmpeg -framerate 30 -i img%01d.png\
          -c:v libx264 -pix_fmt yuv420p out.mp4")
os.system("rm *.png")
