"""Functions for domain coloring plots of complex functions."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def hsl_to_rgb(hsl):
    """
    Convert hsl values to rgb.

    Parameters
    ----------
    hsl : (..., 3) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    rgb : (..., 3) ndarray
       Colors converted to RGB values in range [0, 1]
    """
    hsl = np.asarray(hsl)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsl.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {shp} was found.".format(shp=hsl.shape))

    in_shape = hsl.shape
    hsl = np.array(
        hsl, copy=False,
        dtype=np.promote_types(hsl.dtype, np.float32),  # Don't work on ints.
        ndmin=2,  # In case input was 1D.
    )

    h = hsl[..., 0]
    s = hsl[..., 1]
    l = hsl[..., 2]

    m1 = np.empty_like(h)
    m2 = np.empty_like(h)

    ones = np.ones_like(h)

    idx = l <= 0.5
    m2[idx] = l[idx] * (1.0 * ones[idx] + s[idx])

    idx = l > 0.5
    m2[idx] = l[idx] + s[idx] - (l[idx] * s[idx])

    m1 = 2.0 * l - m2

    r = np.copy(m1)
    g = np.copy(m1)
    b = np.copy(m1)

    hue_r = (h + 1.0/3.0) % 1.0
    hue_g = h % 1.0
    hue_b = (h - 1.0/3.0) % 1.0

    idx = hue_r < 2.0/3.0
    r[idx] = m1[idx] + (m2[idx] - m1[idx]) * (2.0/3.0*ones[idx] - hue_r[idx]) * 6.0
    idx = hue_g < 2.0/3.0
    g[idx] = m1[idx] + (m2[idx] - m1[idx]) * (2.0/3.0*ones[idx] - hue_g[idx]) * 6.0
    idx = hue_b < 2.0/3.0
    b[idx] = m1[idx] + (m2[idx] - m1[idx]) * (2.0/3.0*ones[idx] - hue_b[idx]) * 6.0

    idx = hue_r < 1.0/2.0
    r[idx] = m2[idx]
    idx = hue_g < 1.0/2.0
    g[idx] = m2[idx]
    idx = hue_b < 1.0/2.0
    b[idx] = m2[idx]

    idx = hue_r < 1.0/6.0
    r[idx] = m1[idx] + (m2[idx] - m1[idx]) * hue_r[idx] * 6.0
    idx = hue_g < 1.0/6.0
    g[idx] = m1[idx] + (m2[idx] - m1[idx]) * hue_g[idx] * 6.0
    idx = hue_b < 1.0/6.0
    b[idx] = m1[idx] + (m2[idx] - m1[idx]) * hue_b[idx] * 6.0

    idx = s == 0.0
    r[idx] = ones[idx]
    g[idx] = ones[idx]
    b[idx] = ones[idx]

    rgb = np.stack([r, g, b], axis=-1)

    return rgb.reshape(in_shape)


def gen_z(xrange, yrange, N):
    a = yrange[1]
    b = yrange[0]
    c = xrange[0]
    d = xrange[1]
    y, x = np.ogrid[a:b:(N * 1j), c:d:(N * 1j)]
    z = x + 1j * y
    return z


def complex_im(z):
    if z.dtype != complex:
        return z
    phi = np.angle(z, deg=True)
    hue = (phi % 360) / 360
    sat = 0.85 * np.ones_like(hue)
    val = np.ones_like(hue)
    hsv = np.dstack((hue, sat, val))
    return hsv_to_rgb(hsv)


def complex_plot(f, xrange, yrange, N=2**10):
    z = gen_z(xrange, yrange, N)
    im = complex_im(f(z))
    plt.figure(dpi=150)
    plt.imshow(im, extent=(xrange[0], xrange[1], yrange[0], yrange[1]))
    # plt.gca().set_xticklabels(np.array(plt.gca().get_xticks())/1024)
    plt.xlabel(r'$\Re(k)$')
    plt.ylabel(r'$\Im(k)$')
    plt.show()


def complex_save(f, xrange, yrange, fname, N=2**10):
    z = gen_z(xrange, yrange, N)
    im = complex_im(f(z))
    plt.figure(dpi=150)
    plt.imshow(im, extent=(xrange[0], xrange[1], yrange[0], yrange[1]))
    # plt.gca().set_xticklabels(np.array(plt.gca().get_xticks())/1024)
    plt.xlabel(r'$\Re(k)$')
    plt.ylabel(r'$\Im(k)$')
    plt.savefig(fname)


def lightness(x):
    if x <= 1:
        return np.sqrt(x) / 2
    else:
        return 1 - 8/((x+3)**2)


def complex_im2(z, a):
    if z.dtype != complex:
        return z
    r = np.abs(z)/1e0
    ones = np.ones_like(r)
    phi = np.angle(z, deg=True)
    hue = (phi % 360) / 360
    sat = ones
    lig = ones - np.power(a * ones, r)
    # lig = ones*0.5
    hsl = np.dstack((hue, sat, lig))
    return hsl_to_rgb(hsl)


def complex_plot2(f, xrange, yrange, a=0.999, N=2**10):
    z = gen_z(xrange, yrange, N)
    im = complex_im2(f(z), a)
    plt.figure(dpi=150)
    plt.imshow(im, extent=(xrange[0], xrange[1], yrange[0], yrange[1]))
    # plt.gca().set_xticklabels(np.array(plt.gca().get_xticks())/1024)
    plt.xlabel(r'$\Re(k)$')
    plt.ylabel(r'$\Im(k)$')
    plt.show()


def complex_im3(z, a):
    if z.dtype != complex:
        return z
    r = np.log10(np.abs(z))
    ones = np.ones_like(r)
    phi = np.angle(z, deg=True)
    hue = (phi % 360) / 360
    sat = ones
#     lig = ones - np.power(a * ones, r)
    lig = r
    hsl = np.dstack((hue, sat, lig))
    return hsl_to_rgb(hsl)


def complex_plot3(f, xrange, yrange, a=0.999, N=2**10):
    z = gen_z(xrange, yrange, N)
    im = complex_im3(f(z), a)
    plt.figure(dpi=150)
    plt.imshow(im, extent=(xrange[0], xrange[1], yrange[0], yrange[1]))
    # plt.gca().set_xticklabels(np.array(plt.gca().get_xticks())/1024)
    plt.xlabel(r'$\Re(k)$')
    plt.ylabel(r'$\Im(k)$')
    plt.show()
