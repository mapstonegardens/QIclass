# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------------
"""
Utility functions.

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
import functools
import inspect
import warnings
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

def rms(a):
    """
    Calculates the RMS of an array.

    :param a: An array.

    :returns: The RMS of the array.

    """

    return np.sqrt(np.sum(a**2.0)/a.size)


def moving_average(a, length, mode='valid'):
    """
    Computes the mean in a moving window. Naive implementation.

    Example:
        >>> test = np.array([1,9,9,9,9,9,9,2,3,9,2,2,3,1,1,1,1,3,4,9,9,9,8,3])
        >>> moving_average(test, 7, mode='same')
        [ 4.4285714  5.571428  6.7142857  7.8571428  8.          7.1428571
          7.1428571  6.142857  5.1428571  4.2857142  3.1428571  3.
          2.7142857  1.571428  1.7142857  2.          2.857142  4.
          5.1428571  6.142857  6.4285714  6.1428571  5.7142857  4.5714285 ]

    TODO:
        Other types of average.

    """
    length = int(length)
    pad = int(np.floor(length/2))

    if mode == 'full':
        pad *= 2

    # Make a padded version, paddding with first and last values
    r = np.pad(a, pad, mode='edge')

    # Cumsum with shifting trick; first remove NaNs
    r[np.isnan(r)] = 0
    s = np.cumsum(r, dtype=float)
    s[length:] = s[length:] - s[:-length]
    out = s[length-1:]/length

    # Decide what to return
    if mode == 'same':
        if out.shape[0] != a.shape[0]:
            # If size doesn't match, then interpolate.
            out = (out[:-1, ...] + out[1:, ...]) / 2
        return out
    elif mode == 'valid':
        return out[pad:-pad]
    else:  # mode=='full' and we used a double pad
        return out


def moving_avg_conv(a, length):
    """
    Moving average via convolution. Seems slower than naive.

    """
    boxcar = np.ones(length)/length
    return np.convolve(a, boxcar, mode="same")


def moving_avg_fft(a, length):
    """
    Moving average via FFT convolution. Seems slower than naive.

    """
    boxcar = np.ones(length)/length
    return scipy.signal.fftconvolve(a, boxcar, mode="same")


def normalize(a, new_min=0.0, new_max=1.0):
    """
    Normalize an array to [0,1] or to
    arbitrary new min and max.

    :param a: An array.
    :param new_min: A float to be the new min, default 0.
    :param new_max: A float to be the new max, default 1.

    :returns: The normalized array.
    """

    n = (a - np.amin(a)) / np.amax(a - np.amin(a))
    return n * (new_max - new_min) + new_min


def nearest(a, num):
    """
    Finds the array's nearest value to a given num.
    """
    return a.flat[np.abs(a - num).argmin()]


def next_pow2(num):
    """
    Calculates the next nearest power of 2 to the input. Uses
      2**ceil( log2( num ) ).

    :param num: The number to round to the next power if two.

    :returns: the next power of 2 closest to num.
    """

    return int(2**np.ceil(np.log2(num)))


def top_and_tail(*arrays):
    """
    Top and tail all arrays to the non-NaN extent of the first array.

    E.g. crop the NaNs from the top and tail of a well log.

    """
    if len(arrays) > 1:
        for arr in arrays[1:]:
            assert len(arr) == len(arrays[0])
    nans = np.where(~np.isnan(arrays[0]))[0]
    first, last = nans[0], nans[-1]
    ret_arrays = []
    for array in arrays:
        ret_arrays.append(array[first:last+1])
    return ret_arrays


def extrapolate(a):
    """
    Extrapolate up and down an array from the first and last non-NaN samples.

    E.g. Continue the first and last non-NaN values of a log up and down.

    """
    nans = np.where(~np.isnan(a))[0]
    first, last = nans[0], nans[-1]
    a[:first] = a[first]
    a[last + 1:] = a[last]
    return a

#------------------------------------------------------------------------------------------------

'''
===================
moduli.py
===================

Converts between various acoustic/eslatic parameters,
and provides a way to calculate all the elastic moduli
from Vp, Vs, and rho

Created June 2014

@author: Matt Hall

Using equations http://www.subsurfwiki.org/wiki/Elastic_modulus
from Mavko, G, T Mukerji and J Dvorkin (2003), The Rock Physics Handbook,
Cambridge University Press

'''
import numpy as np

def youngs(vp=None, vs=None, rho=None, mu=None, lam=None, bulk=None, pr=None,
           pmod=None):
    '''
    Computes Young's modulus given either Vp, Vs, and rho, or
    any two elastic moduli (e.g. lambda and mu, or bulk and P
    moduli).

    SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, mu, bulk, pr, and pmod

    Returns:
        Young's modulus in pascals, Pa

    '''
    if (vp is not None) and (vs is not None) and (rho is not None):
        return rho * vs**2 * (3.*vp**2 - 4.*vs**2) / (vp**2 - vs**2)

    elif (mu is not None) and (lam is not None):
        return mu * (3.*lam + 2*mu) / (lam + mu)

    elif (bulk is not None) and (lam is not None):
        return 9.*bulk * (bulk - lam) / (3.*bulk - lam)

    elif (bulk is not None) and (mu is not None):
        return 9.*bulk*mu / (3.*bulk + mu)

    elif (lam is not None) and (pr is not None):
        return lam * (1+pr) * (1 - 2*pr) / pr

    elif (pr is not None) and (mu is not None):
        return 2. * mu * (1+pr)

    elif (pr is not None) and (bulk is not None):
        return 3. * bulk * (1 - 2*pr)

    else:
        return None


def bulk(vp=None, vs=None, rho=None, mu=None, lam=None, youngs=None, pr=None,
         pmod=None):
    '''
    Computes bulk modulus given either Vp, Vs, and rho, or
    any two elastic moduli (e.g. lambda and mu, or Young's
    and P moduli).

    SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, mu, youngs, pr, and pmod

    Returns:
        Bulk modulus in pascals, Pa

    '''

    if (vp is not None) and (vs is not None) and (rho is not None):
        return rho * (vp**2 - (4./3.)*(vs**2))

    elif (mu is not None) and (lam is not None):
        return lam + 2*mu/3.

    elif (mu is not None) and (youngs is not None):
        return youngs * mu / (9.*mu - 3.*youngs)

    elif (lam is not None) and (pr is not None):
        return lam * (1+pr) / 3.*pr

    elif (pr is not None) and (mu is not None):
        return 2. * mu * (1+pr) / (3. - 6.*pr)

    elif (pr is not None) and (youngs is not None):
        return youngs / (3. - 6.*pr)

    elif (lam is not None) and (youngs is not None):
        # Note that this returns a tuple.
        x = np.sqrt(9*lam**2 + 2*youngs*lam + youngs**2)

        def b(y): return 1/6. * (3*lam + youngs + y)

        # Strictly, we should return b(x), b(-x)
        # But actually, the answer is:
        return b(x)

    else:
        return None


def pr(vp=None, vs=None, rho=None, mu=None, lam=None, youngs=None, bulk=None,
       pmod=None):
    '''
    Computes Poisson ratio given either Vp, Vs, and rho, or
    any two elastic moduli (e.g. lambda and mu, or Young's
    and P moduli).

    SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, mu, youngs, bulk, and pmod

    Returns:
        Poisson's ratio, dimensionless

    '''

    if (vp is not None) and (vs is not None):
        return (vp**2. - 2.*vs**2) / (2. * (vp**2 - vs**2))

    elif (mu is not None) and (lam is not None):
        return lam / (2. * (lam+mu))

    elif (mu is not None) and (youngs is not None):
        return (youngs / (2.*mu)) - 1

    elif (lam is not None) and (bulk is not None):
        return lam / (3.*bulk - lam)

    elif (bulk is not None) and (mu is not None):
        return (3.*bulk - 2*mu) / (6.*bulk + 2*mu)

    elif (bulk is not None) and (youngs is not None):
        return (3.*bulk - youngs) / (6.*bulk)

    elif (lam is not None) and (youngs is not None):
        # Note that this returns a tuple.
        x = np.sqrt(9*lam**2 + 2*youngs*lam + youngs**2)

        def b(y): return (1/(4*lam)) * (-1*lam - youngs + y)

        # Strictly, we should return b(x), b(-x)
        # But actually, the answer is:
        return b(x)

    else:
        return None

def lam(vp=None, vs=None, rho=None, pr=None,  mu=None, youngs=None, bulk=None,
        pmod=None):
    '''
    Computes lambda given either Vp, Vs, and rho, or
    any two elastic moduli (e.g. bulk and mu, or Young's
    and P moduli).

    SI units only.

    Args:
        vp, vs, and rho
        or any 2 from bulk, mu, youngs, pr, and pmod

    Returns:
        Lambda in pascals, Pa

    '''
    if (vp is not None) and (vs is not None) and (rho is not None):
        return rho * (vp**2 - 2.*vs**2.)

    elif (youngs is not None) and (mu is not None):
        return mu * (youngs - 2.*mu) / (3.*mu - youngs)

    elif (bulk is not None) and (mu is not None):
        return bulk - (2.*mu/3.)

    elif (bulk is not None) and (youngs is not None):
        return 3. * bulk * (3*bulk - youngs) / (9*bulk - youngs)

    elif (pr is not None) and (mu is not None):
        return 2. * pr * mu / (1 - 2.*pr)

    elif (pr is not None) and (youngs is not None):
        return pr * youngs / ((1+pr) * (1-2*pr))

    elif (pr is not None) and (bulk is not None):
        return 3. * bulk * pr / (1+pr)

    else:
        return None


def mu(vp=None, vs=None, rho=None, pr=None, lam=None, youngs=None, bulk=None,
       pmod=None):
    '''
    Computes shear modulus given either Vp, Vs, and rho, or
    any two elastic moduli (e.g. lambda and bulk, or Young's
    and P moduli).

    SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, bulk, youngs, pr, and pmod

    Returns:
        Shear modulus in pascals, Pa

    '''

    if (vs is not None) and (rho is not None):
        return rho * vs**2

    elif (bulk is not None) and (lam is not None):
        return 3. * (bulk - lam) / 2.

    elif (bulk is not None) and (youngs is not None):
        return 3. * bulk * youngs / (9.*bulk - youngs)

    elif (lam is not None) and (pr is not None):
        return lam * (1 - 2.*pr) / (2.*pr)

    elif (pr is not None) and (youngs is not None):
        return youngs / (2. * (1 + pr))

    elif (pr is not None) and (bulk is not None):
        return 3. * bulk * (1 - 2*pr) / (2. * (1 + pr))

    elif (lam is not None) and (youngs is not None):
        # Note that this returns a tuple.
        x = np.sqrt(9*lam**2 + 2*youngs*lam + youngs**2)

        def b(y): return 1/4. * (-3*lam + youngs + y)

        # Strictly, we should return b(x), b(-x)
        # But actually, the answer is:
        return b(x)

    else:
        return None

def pmod(vp=None, vs=None, rho=None, pr=None, mu=None, lam=None, youngs=None,
         bulk=None):
    '''
    Computes P-wave modulus given either Vp, Vs, and rho, or
    any two elastic moduli (e.g. lambda and mu, or Young's
    and bulk moduli).

    SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, mu, youngs, pr, and bulk

    Returns:
        P-wave modulus in pascals, Pa

    '''

    if (vp is not None) and (rho is not None):
        return rho * vp**2

    elif (lam is not None) and (mu is not None):
        return lam + 2*mu

    elif (youngs is not None) and (mu is not None):
        return mu * (4.*mu - youngs) / (3.*mu - youngs)

    elif (bulk is not None) and (lam is not None):
        return 3*bulk - 2.*lam

    elif (bulk is not None) and (mu is not None):
        return bulk + (4.*mu/3.)

    elif (bulk is not None) and (youngs is not None):
        return 3. * bulk * (3*bulk + youngs) / (9*bulk - youngs)

    elif (lam is not None) and (pr is not None):
        return lam * (1 - pr) / pr

    elif (pr is not None) and (mu is not None):
        return 2. * pr * mu * (1-pr) / (1 - 2.*pr)

    elif (pr is not None) and (youngs is not None):
        return (1-pr) * youngs / ((1+pr) * (1 - 2.*pr))

    elif (pr is not None) and (bulk is not None):
        return 3. * bulk * (1-pr) / (1+pr)

    elif (lam is not None) and (youngs is not None):
        # Note that this returns a tuple.
        x = np.sqrt(9*lam**2 + 2*youngs*lam + youngs**2)

        def b(y): return 1/2. * (-1*lam + youngs + y)

        # Strictly, we should return b(x), b(-x)
        # But actually, the answer is:
        return b(x)

    else:
        return None


def vp(youngs=None, vs=None, rho=None, mu=None, lam=None, bulk=None, pr=None,
       pmod=None):
    '''
    Computes Vp given bulk density and any two elastic moduli
    (e.g. lambda and mu, or Young's and P moduli).

    SI units only.

    Args:
        Any 2 from lam, mu, youngs, pr, pmod, bulk
        Rho

    Returns:
        Vp in m/s

    '''

    if (mu is not None) and (lam is not None) and (rho is not None):
        return np.sqrt((lam + 2.*mu) / rho)

    elif (youngs is not None) and (mu and rho is not None):
        return np.sqrt(mu * (youngs - 4.*mu) / (rho * (youngs - 3.*mu)))

    elif (youngs is not None) and (pr and rho is not None):
        return np.sqrt(youngs * (1 - pr) / (rho * (1+pr) * (1 - 2.*pr)))

    elif (bulk is not None) and (lam and rho is not None):
        return np.sqrt((9.*bulk - 2.*lam) / rho)

    elif (bulk is not None) and (mu is not None and rho is not None):
        return np.sqrt((bulk + 4.*mu/3.) / rho)

    elif (lam is not None) and (pr and rho is not None):
        return np.sqrt(lam * (1. - pr) / (pr*rho))

    else:
        return None


def vs(youngs=None, vp=None, rho=None, mu=None, lam=None, bulk=None, pr=None,
       pmod=None):
    '''
    Computes Vs given bulk density and shear modulus.

    SI units only.

    Args:
        Mu
        Rho

    Returns:
        Vs in m/s

    '''

    if (mu is not None) and (rho is not None):
        return np.sqrt(mu / rho)

    else:
        return None


def moduli_dict(vp, vs, rho):
    '''
    Computes elastic moduli given Vp, Vs, and rho.

    SI units only.

    Args:
        Vp, Vs, and rho

    Returns:
        A dict of elastic moduli, plus P-wave impedance.

    '''

    mod = {}

    mod['imp'] = vp * rho

    mod['mu'] = mu(vs=vs, rho=rho)
    mod['pr'] = pr(vp=vp, vs=vs, rho=rho)
    mod['lam'] = lam(vp=vp, vs=vs, rho=rho)
    mod['bulk'] = bulk(vp=vp, vs=vs, rho=rho)
    mod['pmod'] = pmod(vp=vp, rho=rho)
    mod['youngs'] = youngs(vp=vp, vs=vs, rho=rho)

    return mod
#------------------------------------------------------------------------------------------------
"""
Seismic wavelets.

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
from collections import namedtuple
import numpy as np
from scipy.signal import hilbert
from scipy.signal import chirp

def ricker(duration, dt, f, return_t=False):
    """
    Also known as the mexican hat wavelet, models the function:
    A =  (1-2 \pi^2 f^2 t^2) e^{-\pi^2 f^2 t^2}

    :param duration: The length in seconds of the wavelet.
    :param dt: is the sample interval in seconds (usually 0.001,
               0.002, 0.004)
    :params f: Center frequency of the wavelet (in Hz). If a list or tuple is
               passed, the first element will be used.
    :params return_t: If True, then the function returns a tuple of
                      wavelet, time-basis, where time is the range from
                      -durection/2 to duration/2 in steps of dt.

    :returns: ricker wavelets with center frequency f sampled at t.
    """

    freq = np.array(f)

    t = np.arange(-duration/2, duration/2, dt)

    output = np.zeros((t.size, freq.size))

    for i in range(freq.size):
        pi2 = (np.pi ** 2.0)
        if (freq.size == 1):
            fsqr = freq ** 2.0
        else:
            fsqr = freq[i] ** 2.0
        tsqr = t ** 2.0
        pft = pi2 * fsqr * tsqr
        A = (1 - (2 * pft)) * np.exp(-pft)
        output[:, i] = A

    if freq.size == 1:
        output = output.flatten()

    if return_t:
        RickerWavelet = namedtuple('RickerWavelet', ['amplitude', 'time'])
        return RickerWavelet(output, t)
    else:
        return output


def ormsby(duration, dt, f, return_t=False):
    """
    The Ormsby wavelet requires four frequencies:
    f1 = low-cut frequency
    f2 = low-pass frequency
    f3 = high-pass frequency
    f4 = hi-cut frequency
    Together, the frequencies define a trapezoid shape in the
    spectrum.
    The Ormsby wavelet has several sidelobes, unlike Ricker wavelets
    which only have two, one either side.

    :param duration: The length in seconds of the wavelet.
    :param dt: is the sample interval in seconds (usually 0.001,
               0.002, 0.004)
    :params f: Tuple of form (f1,f2,f3,f4), or a similar list.

    :returns: A vector containing the ormsby wavelet
    """

    # Try to handle some duck typing
    if not (isinstance(f, list) or isinstance(f, tuple)):
        f = [f]

    # Deal with having fewer than 4 frequencies
    if len(f) == 4:
        f1 = f[0]
        f2 = f[1]
        f3 = f[2]
        f4 = f[3]
    else:
        # Cope with only having one frequency
        # This is an arbitrary hack, is this desirable?
        # Need a way to notify with warnings
        f1 = f[0]/4
        f2 = f[0]/2
        f3 = f[0]*2
        f4 = f[0]*2.5

    def numerator(f, t):
        return (np.sinc(f * t)**2) * ((np.pi * f) ** 2)

    pf43 = (np.pi * f4) - (np.pi * f3)
    pf21 = (np.pi * f2) - (np.pi * f1)

    t = np.arange(-duration/2, duration/2, dt)

    A = ((numerator(f4, t)/pf43) - (numerator(f3, t)/pf43) -
         (numerator(f2, t)/pf21) + (numerator(f1, t)/pf21))

    A /= np.amax(A)

    if return_t:
        OrmsbyWavelet = namedtuple('OrmsbyWavelet', ['amplitude', 'time'])
        return OrmsbyWavelet(A, t)
    else:
        return A


def rotate_phase(w, phi, degrees=False):
    """
    Performs a phase rotation of wavelet using:

    The analytic signal can be written in the form S(t) = A(t)exp(j*theta(t))
    where A(t) = magnitude(hilbert(w(t))) and theta(t) = angle(hilbert(w(t))
    then a constant phase rotation phi would produce the analytic signal
    S(t) = A(t)exp(j*(theta(t) + phi)). To get the non analytic signal
    we take real(S(t)) == A(t)cos(theta(t) + phi)
    == A(t)(cos(theta(t))cos(phi) - sin(theta(t))sin(phi)) <= trig idenity
    == w(t)cos(phi) - h(t)sin(phi)

    A = w(t)Cos(phi) - h(t)Sin(phi)
    Where w(t) is the wavelet and h(t) is it's hilbert transform.

    :params w: The wavelet vector.
    :params phi: The phase rotation angle (in radians) to apply.
    :params degrees: Boolean, if phi is in degrees not radians.

    :returns: The phase rotated signal.
    """
    if degrees:
        phi = phi * np.pi / 180.0

    # Get the analytic signal for the wavelet
    a = hilbert(w, axis=0)

    A = (np.real(a) * np.cos(phi) -
         np.imag(a) * np.sin(phi))

    return A
#------------------------------------------------------------------------------------------------
"""
Anisotropy effects.

Backus anisotropy is from thin layers.

Hudson anisotropy is from crack defects.

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
from collections import namedtuple
import numpy as np

def backus_parameters(vp, vs, rho, lb, dz):
    """
    Intermediate parameters for Backus averaging. This is expected to be a
    private function. You probably want backus() and not this.

    Args:
        vp (ndarray): P-wave interval velocity.
        vs (ndarray): S-wave interval velocity.
        rho (ndarray): Bulk density.
        lb (float): The Backus averaging length in m.
        dz (float): The depth sample interval in m.

    Returns:
        tuple: Liner's 5 intermediate parameters: A, C, F, L and M.

    Notes:
        Liner, C (2014), Long-wave elastic attenuation produced by horizontal
        layering. The Leading Edge, June 2014, p 634-638.

    """
    lam1 = lam(vp, vs, rho)
    mu1 = mu(vp, vs, rho)

    # Compute the layer parameters from Liner (2014) equation 2:
    a = rho * np.power(vp, 2.0)  # Acoustic impedance

    # Compute the Backus parameters from Liner (2014) equation 4:
    A1 = 4 * moving_average(mu1*(lam1+mu1)/a, lb/dz, mode='same')
    A = A1 + np.power(moving_average(lam1/a, lb/dz, mode='same'), 2.0)\
        / moving_average(1.0/a, lb/dz, mode='same')
    C = 1.0 / moving_average(1.0/a, lb/dz, mode='same')
    F = moving_average(lam1/a, lb/dz, mode='same')\
        / moving_average(1.0/a, lb/dz, mode='same')
    L = 1.0 / moving_average(1.0/mu1, lb/dz, mode='same')
    M = moving_average(mu1, lb/dz, mode='same')

    BackusResult = namedtuple('BackusResult', ['A', 'C', 'F', 'L', 'M'])
    return BackusResult(A, C, F, L, M)


def backus(vp, vs, rho, lb, dz):
    """
    Backus averaging. Using Liner's algorithm (2014; see Notes).

    Args:
        vp (ndarray): P-wave interval velocity.
        vs (ndarray): S-wave interval velocity.
        rho (ndarray): Bulk density.
        lb (float): The Backus averaging length in m.
        dz (float): The depth sample interval in m.

    Returns:
        namedtuple: the smoothed logs: vp, vs, plus rho. Useful for computing
            other elastic parameters at a seismic scale.

    Notes:
        Liner, C (2014), Long-wave elastic attenuation produced by horizontal
        layering. The Leading Edge, June 2014, p 634-638.

    """
    # Compute the Backus parameters:
    A, C, F, L, M = backus_parameters(vp, vs, rho, lb, dz)

    # Compute the vertical velocities from Liner (2014) equation 5:
    R = moving_average(rho, lb/dz, mode='same')
    vp0 = np.sqrt(C / R)
    vs0 = np.sqrt(L / R)

    BackusResult = namedtuple('BackusResult', ['Vp', 'Vs', 'rho'])
    return BackusResult(Vp=vp0, Vs=vs0, rho=R)


def backus_quality_factor(vp, vs, rho, lb, dz):
    """
    Compute Qp and Qs from Liner (2014) equation 10.

    """
    vp0, vs0, _ = backus(vp, vs, rho, lb, dz)

    ptemp = np.pi * np.log(vp0 / vp) / (np.log(vp0 / vp) + np.log(lb/dz))
    Qp = 1.0 / np.tan(ptemp)

    stemp = np.pi * np.log(vs0 / vs) / (np.log(vs0 / vs) + np.log(lb/dz))
    Qs = 1.0 / np.tan(stemp)

    BackusResult = namedtuple('BackusResult', ['Qp', 'Qs'])
    return BackusResult(Qp=Qp, Qs=Qs)


def thomsen_parameters(vp, vs, rho, lb, dz):
    """
    Liner, C, and T Fei (2006). Layer-induced seismic anisotropy from
    full-wave sonic logs: Theory, application, and validation.
    Geophysics 71 (6), p D183–D190. DOI:10.1190/1.2356997

    """
    A, C, F, L, M = backus_parameters(vp, vs, rho, lb, dz)

    delta = ((F + L)**2.0 - (C - L)**2.0) / (2.0 * C * (C - L))
    epsilon = (A - C) / (2.0 * C)
    gamma = (M - L) / (2.0 * L)

    ThomsenParameters = namedtuple('ThomsenParameters', ['δ', 'ε', 'γ'])
    return ThomsenParameters(delta, epsilon, gamma)


def dispersion_parameter(qp):
    """
    Kjartansson (1979). Journal of Geophysical Research, 84 (B9),
    4737-4748. DOI: 10.1029/JB084iB09p04737.
    """
    return np.arctan(1/qp) / np.pi


def blangy(vp1, vs1, rho1, d1, e1, vp0, vs0, rho0, d0, e0, theta):
    """
    Blangy, JP, 1994, AVO in transversely isotropic media-An overview.
    Geophysics 59 (5), 775-781. DOI: 10.1190/1.1443635

    Provide Vp, Vs, rho, delta, epsilon for the upper and lower intervals,
    and theta, the incidence angle.

    :param vp1: The p-wave velocity of the upper medium.
    :param vs1: The s-wave velocity of the upper medium.
    :param rho1: The density of the upper medium.
    :param d1: Thomsen's delta of the upper medium.
    :param e1: Thomsen's epsilon of the upper medium.

    :param vp0: The p-wave velocity of the lower medium.
    :param vs0: The s-wave velocity of the lower medium.
    :param rho0: The density of the lower medium.
    :param d0: Thomsen's delta of the lower medium.
    :param e0: Thomsen's epsilon of the lower medium.

    :param theta: A scalar [degrees].

    :returns: the isotropic and anisotropic reflectivities in a tuple. The
        isotropic result is equivalent to Aki-Richards.


    TODO
        Use rocks.
    """
    lower = {'vp': vp0,
             'vs': vs0,
             'rho': rho0,
             'd': d0,
             'e': e0,
             }

    upper = {'vp': vp1,
             'vs': vs1,
             'rho': rho1,
             'd': d1,
             'e': e1,
             }

    # Redefine theta
    inc_angle = np.radians(theta)
    trans_angle = np.arcsin(np.sin(inc_angle) * lower['vp']/upper['vp'])
    theta = 0.5 * (inc_angle + trans_angle)

    vp = (upper['vp'] + lower['vp'])/2.0
    vs = (upper['vs'] + lower['vs'])/2.0
    rho = (upper['rho'] + lower['rho'])/2.0

    dvp = lower['vp'] - upper['vp']
    dvs = lower['vs'] - upper['vs']
    drho = lower['rho'] - upper['rho']
    dd = lower['d'] - upper['d']
    de = lower['e'] - upper['e']

    A = 0.5 * (drho/rho + dvp/vp)
    B = 2.0 * (vs**2 / vp**2) * ((drho/rho + 2 * dvs/vs)) * np.sin(theta)**2
    C = 0.5 * (dvp/vp) * np.tan(theta)**2
    D = 0.5 * dd * np.sin(theta)**2
    E = 0.5 * (dd - de) * np.sin(theta)**2 * np.tan(theta)**2

    isotropic = A - B + C
    anisotropic = isotropic + D - E

    BlangyResult = namedtuple('BlangyResult', ['isotropic', 'anisotropic'])
    return BlangyResult(isotropic, anisotropic)


def ruger(vp1, vs1, rho1, d1, e1, vp2, vs2, rho2, d2, e2, theta):
    """
    Coded by Alessandro Amato del Monte and (c) 2016 by him
    https://github.com/aadm/avo_explorer/blob/master/avo_explorer_v2.ipynb

    Rüger, A., 1997, P -wave reflection coefficients for transversely
    isotropic models with vertical and horizontal axis of symmetry:
    Geophysics, v. 62, no. 3, p. 713–722.

    Provide Vp, Vs, rho, delta, epsilon for the upper and lower intervals,
    and theta, the incidence angle.

    :param vp1: The p-wave velocity of the upper medium.
    :param vs1: The s-wave velocity of the upper medium.
    :param rho1: The density of the upper medium.
    :param d1: Thomsen's delta of the upper medium.
    :param e1: Thomsen's epsilon of the upper medium.

    :param vp0: The p-wave velocity of the lower medium.
    :param vs0: The s-wave velocity of the lower medium.
    :param rho0: The density of the lower medium.
    :param d0: Thomsen's delta of the lower medium.
    :param e0: Thomsen's epsilon of the lower medium.

    :param theta: A scalar [degrees].

    :returns: anisotropic reflectivity.

    """
    a = np.radians(theta)
    vp = np.mean([vp1, vp2])
    vs = np.mean([vs1, vs2])
    z = np.mean([vp1*rho1, vp2*rho2])
    g = np.mean([rho1*vs1**2, rho2*vs2**2])
    dvp = vp2-vp1
    z2, z1 = vp2*rho2, vp1*rho1
    dz = z2-z1
    dg = rho2*vs2**2 - rho1*vs1**2
    dd = d2-d1
    de = e2-e1
    A = 0.5*(dz/z)
    B = 0.5*(dvp/vp - (2*vs/vp)**2 * (dg/g) + dd) * np.sin(a)**2
    C = 0.5*(dvp/vp + de) * np.sin(a)**2 * np.tan(a)**2
    R = A+B+C

    return R


def crack_density(porosity, aspect):
    """
    Returns crack density from porosity and aspect ratio, phi and alpha
    respectively in the unnumbered equation between 15.40 and 15.41 in
    Dvorkin et al. 2014.

    Args:
        porosity (float): Fractional porosity.
        aspect (float): Aspect ratio.

    Returns:
        float: Crack density.
    """
    if porosity >= 1:
        porosity /= 100.

    return 3 * porosity / (4 * np.pi * aspect)


def hudson_delta_M(porosity, aspect, mu, lam=None, pmod=None):
    """
    The approximate reduction in compressional modulus M in the direction
    normal to a set of aligned cracks. Eqn 15.40 in Dvorkin et al (2014).

    Args:
        porosity (float): Fractional porosity, phi.
        aspect (float): Aspect ratio, alpha.
        mu (float): Shear modulus, sometimes called G.
        lam (float): Lame's first parameter.
        pmod (float): Compressional modulus, M.

    Returns:
        float: M_inf - M_0 = \Delta c_11.
    """
    epsilon = crack_density(porosity, aspect)
    if lam:
        return epsilon * (lam**2 / mu) * 4*(lam + 2*mu)/(3*lam + 3*mu)
    else:
        return (4*epsilon/3) * ((pmod - 2*mu)**2 / mu) * (pmod/(pmod-mu))


def hudson_delta_G(porosity, aspect, mu, lam=None, pmod=None):
    """
    The approximate reduction in shear modulus G (or mu) in the direction
    normal to a set of aligned cracks. Eqn 15.42 in Dvorkin et al (2014).

    Args:
        porosity (float): Fractional porosity, phi.
        aspect (float): Aspect ratio, alpha.
        mu (float): Shear modulus, sometimes called G.
        lam (float): Lame's first parameter, lambda.
        pmod (float): Compressional modulus, M.

    Returns:
        float: M_inf - M_0 = \Delta c_11.
    """
    epsilon = crack_density(porosity, aspect)
    if lam:
        return epsilon * mu * 16*(lam + 2*mu)/(9*lam + 12*mu)
    else:
        return (16*mu*epsilon/3) * pmod / (3*pmod - 2*mu)


def hudson_quality_factor(porosity, aspect, mu, lam=None, pmod=None):
    """
    Returns Q_p and Q_s for cracked media. Equations 15.41 and 15.43 in
    Dvorkin et al. (2014).
    """
    Qp = 2*mu / hudson_delta_M(porosity, aspect, mu, lam, pmod)
    Qs = 2*mu / hudson_delta_G(porosity, aspect, mu, lam, pmod)
    return Qp, Qs


def hudson_inverse_Q_ratio(mu=None,
                           pmod=None,
                           pr=None,
                           vp=None,
                           vs=None,
                           aligned=True):
    """
    Dvorkin et al. (2014), Eq 15.44 (aligned) and 15.48 (not aligned).
    """
    if pr:
        x = (2 - 2*pr) / (1 - 2*pr)
    elif vp and vs:
        x = vp**2 / vs**2
    elif mu and pmod:
        x = pmod / mu
    else:
        raise Exception

    if aligned:
        return 0.25 * (x - 2)**2 * (3*x - 2) / (x**2 - x)
    else:
        a = 2*x / (3*x - 2)
        b = x / 3*(x - 1)
        return 1.25 * ((x - 2)**2 / (x - 1)) / (a + b)

#------------------------------------------------------------------------------------------------    
'''
===================
fluidsub.py
===================

Calculates various parameters for fluid substitution
from Vp, Vs, and rho

Created July 2014

@author: Matt Hall, Evan Bianco

Using http://www.subsurfwiki.org/wiki/Gassmann_equation

The algorithm is from Avseth et al (2006), per the wiki page.

Informed by Smith et al, Geophysics 68(2), 2003.

At some point we should do Biot too, per Russell...
http://cseg.ca/symposium/archives/2012/presentations/Biot_Gassmann_and_me.pdf

'''
from collections import namedtuple
import numpy as np

def avseth_gassmann(ksat1, kf1, kf2, kmin, phi):
    """
    Applies the Gassmann equation. Takes Ksat1,
    Kfluid1, Kfluid2, Kmineral, and phi.

    Returns Ksat2.
    """

    s = ksat1 / (kmin - ksat1)
    f1 = kf1 / (phi * (kmin - kf1))
    f2 = kf2 / (phi * (kmin - kf2))

    ksat2 = kmin / ((1/(s - f1 + f2)) + 1)

    return ksat2


def smith_gassmann(kstar, k0, kfl2, phi):
    """
    Applies the Gassmann equation.

    Returns Ksat2.
    """

    a = (1 - kstar/k0)**2.0
    b = phi/kfl2 + (1-phi)/k0 - (kstar/k0**2.0)

    ksat2 = kstar + (a/b)

    return ksat2


def vrh(kclay, kqtz, vclay):
    """
    Voigt-Reuss-Hill average to find Kmatrix from clay and qtz components.

    From Smith et al, Geophysics 68(2), 2003.

    Works for any two components.

    Returns Kvrh, AKA Kmatrix.
    """

    vqtz = 1 - vclay

    kreuss = 1. / (vclay/kclay + vqtz/kqtz)
    kvoigt = vclay*kclay + vqtz*kqtz
    kvrh = 0.5 * (kreuss + kvoigt)

    return kvrh


def rhogas(gravity, temp, pressure):
    """
    From http://www.spgindia.org/geohorizon/jan_2006/dhananjay_paper.pdf
    """
    R = 8.3144621  # Gas constant in J.mol^-1.K^-1

    # Compute pseudo-reduced temp and pressure:
    tpr = (temp + 273.15) / (gravity * (94.72 + 170.75))
    ppr = pressure / (4.892 - 0.4048*gravity)

    exponent = -1 * (0.45 + 8 * (0.56 - (1/tpr))**2.0) * ppr**1.2 / tpr
    bige = 0.109 * (3.85 - tpr)**2.0 * np.exp(exponent)
    term2 = 0.642*tpr - 0.007*tpr**4.0 - 0.52

    Z = ppr * (0.03 + 0.00527 * (3.5 - tpr)) + term2 + bige

    rhogas = 28.8 * gravity * pressure / (Z * R * (temp + 273.15))

    return rhogas


def rhosat(phi, sw, rhomin, rhow, rhohc):
    """
    Density of partially saturated rock.

    """
    a = rhomin * (1 - phi)        # grains
    b = rhow * sw * phi           # brine
    c = rhohc * (1 - sw) * phi    # hydrocarbon

    return a + b + c


def avseth_fluidsub(vp, vs, rho, phi, rhof1, rhof2, kmin, kf1, kf2):
    """
    Naive fluid substitution from Avseth et al.
    No pressure/temperature correction.

    :param vp: P-wave velocity
    :param vs: S-wave velocity
    :param rho: bulk density
    :param phi: porosity (i.e. 0.20)
    :param rhof1: bulk density of original fluid (base case)
    :param rhof2: bulk density of substitute fluid (subbed case)
    :param kmin: bulk modulus of solid mineral(s)
    :param kf1: bulk modulus of original fluid
    :param kf2: bulk modulus of substitue fluid

    Only works for SI units right now.

    Returns Vp, Vs, and rho for the substituted case
    """

    # Step 1: Extract the dynamic bulk and shear moduli
    ksat1 = bulk(vp=vp, vs=vs, rho=rho)
    musat1 = mu(vp=vp, vs=vs, rho=rho)

    # Step 2: Apply Gassmann's relation
    ksat2 = avseth_gassmann(ksat1=ksat1, kf1=kf1, kf2=kf2, kmin=kmin, phi=phi)

    # Step 3: Leave the shear modulus unchanged
    musat2 = musat1

    # Step 4: Correct the bulk density for the change in fluid
    rho2 = rho + phi * (rhof2 - rhof1)

    # Step 5: recompute the fluid substituted velocities
    vp2 = vp(bulk=ksat2, mu=musat2, rho=rho2)
    vs2 = vs(mu=musat2, rho=rho2)

    FluidSubResult = namedtuple('FluidSubResult', ['Vp', 'Vs', 'rho'])
    return FluidSubResult(vp2, vs2, rho2)


def smith_fluidsub(vp, vs, rho, phi, rhow, rhohc,
                   sw, swnew, kw, khc, kclay, kqtz,
                   vclay=None,
                   rhownew=None, rhohcnew=None,
                   kwnew=None, khcnew=None
                   ):
    """
    Naive fluid substitution from Smith et al. 2003
    No pressure/temperature correction.

    :param vp: P-wave velocity
    :param vs: S-wave velocity

    :param rho: bulk density
    :param rhow: density of water
    :param rhohc: density of HC
    :param rhownew: density of water in subbed case (optional)
    :param rhohcnew: density of HC in subbed case (optional)

    :param phi: porosity (fraction)

    :param sw: water saturation in base case
    :param swnew: water saturation in subbed case

    :param kw:  bulk modulus of water
    :param khc: bulk modulus of hydrocarbon
    :param kwnew:  bulk modulus of water in subbed case (optional)
    :param khcnew: bulk modulus of hydrocarbon in subbed case (optional)

    :param vclay: Vclay (give this or vsh)
    :param vsh: Vsh (or give vclay; vclay = 0.7 * vsh)
    :param kclay: bulk modulus of clay (DEFAULT?)
    :param kqtz:  bulk modulus of quartz (DEFAULT?)

    Only works for SI units right now.

    Returns Vp, Vs, and rho for the substituted case.
    """

    # Using the workflow in Smith et al., Table 2
    # Using Smith's notation, more or less (not the same
    # as Avseth's notation).
    #
    # Step 1: Log edits and interpretation.
    #
    # Step 2. Shear velocity estimation, if necessary.
    #
    # Step 3. Calculate K and G for the in-situ conditions.
    ksat = bulk(vp=vp, vs=vs, rho=rho)
    g = mu(vp=vp, vs=vs, rho=rho)

    # Step 4. Calculate K0 based on lithology estimates (VRH or HS mixing).
    k0 = vrh(kclay=kclay, kqtz=kqtz, vclay=vclay)

    # Step 5. Calculate fluid properties (K and ρ).
    # Step 6. Mix fluids for the in-situ case according to Sw.
    kfl = 1 / (sw/kw + (1-sw)/khc)
    rhofl = sw * rhow + (1-sw)*rhohc

    # Step 7: Calculate K*.
    a = ksat * ((phi*k0/kfl) + 1 - phi) - k0
    b = (phi*k0/kfl) + (ksat/k0) - 1 - phi
    kstar = a / b

    # Step 8: Calculate new fluid properties (K and ρ) at the desired Sw
    # First set the new fluid properties, in case they are unchanged.
    if kwnew is None:
        kwnew = kw
    if rhownew is None:
        rhownew = rhow
    if khcnew is None:
        khcnew = khc
    if rhohcnew is None:
        rhohcnew = rhohc

    # Now calculate the new fluid properties.
    kfl2 = 1 / (swnew/kwnew + (1-swnew)/khcnew)
    rhofl2 = swnew * rhownew + (1-swnew)*rhohcnew

    # Step 9: Calculate the new saturated bulk modulus of the rock
    # using Gassmann.
    ksat2 = smith_gassmann(kstar=kstar, k0=k0, kfl2=kfl2, phi=phi)

    # Step 10: Calculate the new bulk density.
    # First we need the grain density...
    rhog = (rho - phi*rhofl) / (1-phi)
    # Now we can find the new bulk density
    rho2 = phi*rhofl2 + rhog*(1-phi)

    # Step 11: Calculate the new compressional velocity.
    # Remember, mu (G) is unchanged.
    vp2 = vp(bulk=ksat2, mu=g, rho=rho2)

    # Step 12: Calculate the new shear velocity.
    vs2 = vs(mu=g, rho=rho2)

    # Finish.
    FluidSubResult = namedtuple('FluidSubResult', ['Vp', 'Vs', 'rho'])
    return FluidSubResult(vp2, vs2, rho2)

#------------------------------------------------------------------------------------------------
"""
Various reflectivity algorithms.

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
from collections import namedtuple

import numpy as np
from numpy import tan, sin, cos

def scattering_matrix(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    '''
    Full Zoeppritz solution, considered the definitive solution.
    Calculates the angle dependent p-wave reflectivity of an interface
    between two mediums.

    Originally written by: Wes Hamlyn, vectorized by Agile.

    :param vp1: The p-wave velocity of the upper medium.
    :param vs1: The s-wave velocity of the upper medium.
    :param rho1: The density of the upper medium.

    :param vp0: The p-wave velocity of the lower medium.
    :param vs0: The s-wave velocity of the lower medium.
    :param rho0: The density of the lower medium.

    :param theta1: A scalar  [degrees].

    :returns: a 4x4 array representing the scattering matrix
                at the incident angle theta1.
    '''
    # Make sure theta1 is an array.
    theta1 = np.radians(np.array(theta1))
    if theta1.size == 1:
        theta1 = np.expand_dims(theta1, axis=1)

    # Set the ray paramter, p.
    p = sin(theta1) / vp1

    # Calculate reflection & transmission angles for Zoeppritz.
    theta2 = np.arcsin(p * vp2)  # Trans. angle of P-wave.
    phi1 = np.arcsin(p * vs1)    # Refl. angle of converted S-wave.
    phi2 = np.arcsin(p * vs2)    # Trans. angle of converted S-wave.

    # Matrix form of Zoeppritz Equations... M & N are matrices.
    M = np.array([[-sin(theta1), -cos(phi1), sin(theta2), cos(phi2)],
                  [cos(theta1), -sin(phi1), cos(theta2), -sin(phi2)],
                  [2 * rho1 * vs1 * sin(phi1) * cos(theta1),
                   rho1 * vs1 * (1 - 2 * sin(phi1) ** 2),
                   2 * rho2 * vs2 * sin(phi2) * cos(theta2),
                   rho2 * vs2 * (1 - 2 * sin(phi2) ** 2)],
                  [-rho1 * vp1 * (1 - 2 * sin(phi1) ** 2),
                   rho1 * vs1 * sin(2 * phi1),
                   rho2 * vp2 * (1 - 2 * sin(phi2) ** 2),
                   -rho2 * vs2 * sin(2 * phi2)]], dtype='float')

    N = np.array([[sin(theta1), cos(phi1), -sin(theta2), -cos(phi2)],
                  [cos(theta1), -sin(phi1), cos(theta2), -sin(phi2)],
                  [2 * rho1 * vs1 * sin(phi1) * cos(theta1),
                   rho1 * vs1 * (1 - 2 * sin(phi1) ** 2),
                   2 * rho2 * vs2 * sin(phi2) * cos(theta2),
                   rho2 * vs2 * (1 - 2 * sin(phi2) ** 2)],
                  [rho1 * vp1 * (1 - 2 * sin(phi1) ** 2),
                   -rho1 * vs1 * sin(2 * phi1),
                   - rho2 * vp2 * (1 - 2 * sin(phi2) ** 2),
                   rho2 * vs2 * sin(2 * phi2)]], dtype='float')

    A = np.linalg.inv(np.rollaxis(M, 2))
    Z = np.matmul(A, np.rollaxis(N, -1))

    return np.rollaxis(Z, 0, 3)


def zoeppritz_element(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, element='PdPu'):
    """
    Returns any mode reflection coefficients from the Zoeppritz
    scattering matrix. Pass in the mode as element, e.g. 'PdSu' for PS.

    Wraps scattering_matrix().

    :returns: a vector of len(theta1) containing the reflectivity
             value corresponding to each angle.
    """
    elements = np.array([['PdPu', 'SdPu', 'PuPu', 'SuPu'],
                         ['PdSu', 'SdSu', 'PuSu', 'SuSu'],
                         ['PdPd', 'SdPd', 'PuPd', 'SuPd'],
                         ['PdSd', 'SdSd', 'PuSd', 'SuSd']])

    Z = scattering_matrix(vp1, vs1, rho1, vp2, vs2, rho2, theta1)

    return np.squeeze(Z[np.where(elements == element)])


def zoeppritz(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    '''
    Returns the PP reflection coefficients from the Zoeppritz
    scattering matrix.

    Wraps zoeppritz_element().

    :returns: a vector of len(theta1) containing the reflectivity
             value corresponding to each angle.
    '''
    return zoeppritz_element(vp1, vs1, rho1, vp2, vs2, rho2, theta1, 'PdPu')


def zoeppritz_rpp(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    Exact Zoeppritz from expression.

    This is useful because we can pass arrays to it, which we can't do to
    scattering_matrix().

    Dvorkin et al. (2014). Seismic Reflections of Rock Properties. Cambridge.
    """
    theta1 = np.radians(theta1)
    p = np.sin(theta1) / vp1  # Ray parameter
    theta2 = np.arcsin(p * vp2)
    phi1 = np.arcsin(p * vs1)  # Reflected S
    phi2 = np.arcsin(p * vs2)  # Transmitted S

    a = rho2 * (1 - 2 * np.sin(phi2)**2.) - rho1 * (1 - 2 * np.sin(phi1)**2.)
    b = rho2 * (1 - 2 * np.sin(phi2)**2.) + 2 * rho1 * np.sin(phi1)**2.
    c = rho1 * (1 - 2 * np.sin(phi1)**2.) + 2 * rho2 * np.sin(phi2)**2.
    d = 2 * (rho2 * vs2**2 - rho1 * vs1**2)

    E = (b * np.cos(theta1) / vp1) + (c * np.cos(theta2) / vp2)
    F = (b * np.cos(phi1) / vs1) + (c * np.cos(phi2) / vs2)
    G = a - d * np.cos(theta1)/vp1 * np.cos(phi2)/vs2
    H = a - d * np.cos(theta2)/vp2 * np.cos(phi1)/vs1

    D = E*F + G*H*p**2

    rpp = (1/D) * (F*(b*(np.cos(theta1)/vp1) - c*(np.cos(theta2)/vp2)) \
                   - H*p**2 * (a + d*(np.cos(theta1)/vp1)*(np.cos(phi2)/vs2)))

    return rpp


def akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    This is the formulation from Avseth et al.,
    Quantitative seismic interpretation,
    Cambridge University Press, 2006. Adapted for a 4-term formula.
    See http://subsurfwiki.org/wiki/Aki-Richards_equation

    :param vp1: The p-wave velocity of the upper medium.
    :param vs1: The s-wave velocity of the upper medium.
    :param rho1: The density of the upper medium.

    :param vp2: The p-wave velocity of the lower medium.
    :param vs2: The s-wave velocity of the lower medium.
    :param rho2: The density of the lower medium.

    :param theta1: An array of incident angles to use for reflectivity
                   calculation [degrees].

    :returns: a vector of len(theta1) containing the reflectivity
             value corresponding to each angle
    """

    # We are not using this for anything, but will
    # critical_angle = arcsin(vp1/vp2)

    # Do we need to ensure that we get floats out before
    # computing sines?
    if np.ndim(vp1) == 0:
        vp1 = float(vp1)
    else:
        vp1 = np.array(vp1).astype(float)

    theta1 = np.radians(theta1)
    theta2 = np.arcsin(vp2/vp1*sin(theta1))

    # Compute the various parameters
    drho = rho2-rho1
    dvp = vp2-vp1
    dvs = vs2-vs1
    meantheta = (theta1+theta2) / 2.0
    rho = (rho1+rho2) / 2.0
    vp = (vp1+vp2) / 2.0
    vs = (vs1+vs2) / 2.0

    # Compute the coefficients
    w = 0.5 * drho/rho
    x = 2 * (vs/vp1)**2 * drho/rho
    y = 0.5 * (dvp/vp)
    z = 4 * (vs/vp1)**2 * (dvs/vs)

    # Compute the terms
    term1 = w
    term2 = -1 * x * sin(theta1)**2
    term3 = y / cos(meantheta)**2
    term4 = -1 * z * sin(theta1)**2

    if terms:
        fields = ['term1', 'term2', 'term3', 'term4']
        AkiRichards = namedtuple('AkiRichards', fields)
        return AkiRichards(term1, term2, term3, term4)
    else:
        return (term1 + term2 + term3 + term4)


def akirichards_alt(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    This is another formulation of the Aki-Richards solution.
    See http://subsurfwiki.org/wiki/Aki-Richards_equation

    :param vp1: The p-wave velocity of the upper medium.
    :param vs1: The s-wave velocity of the upper medium.
    :param rho1: The density of the upper medium.

    :param vp2: The p-wave velocity of the lower medium.
    :param vs2: The s-wave velocity of the lower medium.
    :param rho2: The density of the lower medium.

    :param theta1: An array of incident angles to use for reflectivity
                   calculation [degrees].

    :returns: a vector of len(theta1) containing the reflectivity
             value corresponding to each angle.
    """

    # We are not using this for anything, but will
    # critical_angle = arcsin(vp1/vp2)

    # Do we need to ensure that we get floats out before
    # computing sines?
    if np.ndim(vp1) == 0:
        vp1 = float(vp1)
    else:
        vp1 = np.array(vp1).astype(float)

    theta1 = np.radians(theta1)
    theta2 = np.arcsin(vp2/vp1*sin(theta1))

    # Compute the various parameters
    drho = rho2-rho1
    dvp = vp2-vp1
    dvs = vs2-vs1
    theta = (theta1+theta2)/2.0
    rho = (rho1+rho2)/2.0
    vp = (vp1+vp2)/2.0
    vs = (vs1+vs2)/2.0

    # Compute the three terms
    term1 = 0.5 * (dvp/vp + drho/rho)
    term2 = (0.5*dvp/vp-2*(vs/vp)**2*(drho/rho+2*dvs/vs)) * sin(theta)**2
    term3 = 0.5 * dvp/vp * (tan(theta)**2 - sin(theta)**2)

    if terms:
        fields = ['term1', 'term2', 'term3']
        AkiRichards = namedtuple('AkiRichards', fields)
        return AkiRichards(term1, term2, term3)
    else:
        return (term1 + term2 + term3)


def fatti(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    Compute reflectivities with Fatti's formulation of the
    Aki-Richards equation, which does not account for the
    critical angle. Fatti et al (1994), Geophysics 59 (9).

    :param vp1: The p-wave velocity of the upper medium.
    :param vs1: The s-wave velocity of the upper medium.
    :param rho1: The density of the upper medium.

    :param vp2: The p-wave velocity of the lower medium.
    :param vs2: The s-wave velocity of the lower medium.
    :param rho2: The density of the lower medium.

    :param theta1: An array of incident angles to use for reflectivity
                   calculation [degrees].

    :returns: a vector of len(theta1) containing the reflectivity
             value corresponding to each angle.
    """
    # Do we need to ensure that we get floats out before computing
    # sines?
    if np.ndim(vp1) == 0:
        vp1 = float(vp1)
    else:
        vp1 = np.array(vp1).astype(float)

    theta1 = np.radians(theta1)

    # Compute the various parameters
    drho = rho2-rho1
    rho = (rho1+rho2) / 2.0
    vp = (vp1+vp2) / 2.0
    vs = (vs1+vs2) / 2.0
    dip = (vp2*rho2 - vp1*rho1)/(vp2*rho2 + vp1*rho1)
    dis = (vs2*rho2 - vs1*rho1)/(vs2*rho2 + vs1*rho1)
    d = drho/rho

    # Compute the three terms
    term1 = (1 + tan(theta1)**2) * dip
    term2 = -8 * (vs/vp)**2 * dis * sin(theta1)**2
    term3 = -1 * (0.5 * tan(theta1)**2 - 2 * (vs/vp)**2 * sin(theta1)**2) * d

    if terms:
        fields = ['term1', 'term2', 'term3']
        Fatti = namedtuple('Fatti', fields)
        return Fatti(term1, term2, term3)
    else:
        return (term1 + term2 + term3)


def shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0,
          terms=False,
          return_gradient=False):
    """
    Compute Shuey approximation with 3 terms.
    http://subsurfwiki.org/wiki/Shuey_equation

    :param vp1: The p-wave velocity of the upper medium.
    :param vs1: The s-wave velocity of the upper medium.
    :param rho1: The density of the upper medium.
    :param vp2: The p-wave velocity of the lower medium.
    :param vs2: The s-wave velocity of the lower medium.
    :param rho2: The density of the lower medium.
    :param theta1: An array of incident angles to use for reflectivity
                   calculation [degrees].
    :param terms: bool. Whether to return a tuple of the 3 individual terms.
    :param return_gradient: bool. Whether to return a tuple of the intercept
                            and gradient (i.e. the second term divided by
                            sin^2(theta).

    :returns: a vector of len(theta1) containing the reflectivity
             value corresponding to each angle.
    """
    theta1 = np.radians(theta1)

    # Compute some parameters
    drho = rho2-rho1
    dvp = vp2-vp1
    dvs = vs2-vs1
    rho = (rho1+rho2)/2.0
    vp = (vp1+vp2)/2.0
    vs = (vs1+vs2)/2.0

    # Compute three-term reflectivity

    r0 = 0.5 * (dvp/vp + drho/rho)
    g = 0.5 * dvp/vp - 2 * (vs**2/vp**2) * (drho/rho + 2 * dvs/vs)
    f = 0.5 * dvp/vp

    term1 = r0
    term2 = g * np.sin(theta1)**2
    term3 = f * (np.tan(theta1)**2 - np.sin(theta1)**2)

    if return_gradient:
        fields = ['intercept', 'gradient']
        Shuey = namedtuple('Shuey', fields)
        return Shuey(r0, g)
    elif terms:
        fields = ['R0', 'Rg', 'Rf']
        Shuey = namedtuple('Shuey', fields)
        return Shuey(term1, term2, term3)
    else:
        return (term1 + term2 + term3)

def bortfeld(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    Compute Bortfeld approximation with three terms.
    http://sepwww.stanford.edu/public/docs/sep111/marie2/paper_html/node2.html

    :param vp1: The p-wave velocity of the upper medium.
    :param vs1: The s-wave velocity of the upper medium.
    :param rho1: The density of the upper medium.

    :param vp2: The p-wave velocity of the lower medium.
    :param vs2: The s-wave velocity of the lower medium.
    :param rho2: The density of the lower medium.

    :param theta1: An array of incident angles to use for reflectivity
                   calculation [degrees].

    :returns: a vector of len(theta1) containing the reflectivity
             value corresponding to each angle.
    """
    theta1 = np.radians(theta1)

    # Compute some parameters
    drho = rho2-rho1
    dvp = vp2-vp1
    dvs = vs2-vs1
    rho = (rho1+rho2)/2.0
    vp = (vp1+vp2)/2.0
    vs = (vs1+vs2)/2.0
    k = (2 * vs/vp)**2
    rsh = 0.5 * (dvp/vp - k*drho/rho - 2*k*dvs/vs)

    # Compute three-term reflectivity
    term1 = 0.5 * (dvp/vp + drho/rho)
    term2 = rsh * np.sin(theta1)**2
    term3 = 0.5 * dvp/vp * np.tan(theta1)**2 * np.sin(theta1)**2

    if terms:
        fields = ['term1', 'term2', 'term3']
        Bortfeld = namedtuple('Bortfeld', fields)
        return Bortfeld(term1, term2, term3)
    else:
        return (term1 + term2 + term3)


def hilterman(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    Not recommended, only seems to match Zoeppritz to about 10 deg.

    Hilterman (1989) approximation from Mavko et al. Rock Physics Handbook.
    According to Dvorkin: "arguably the simplest and a very convenient
    [approximation]." At least for small angles and small contrasts.

    :param vp1: The p-wave velocity of the upper medium.
    :param vs1: The s-wave velocity of the upper medium.
    :param rho1: The density of the upper medium.

    :param vp2: The p-wave velocity of the lower medium.
    :param vs2: The s-wave velocity of the lower medium.
    :param rho2: The density of the lower medium.

    :param theta1: An array of incident angles to use for reflectivity
                   calculation [degrees].

    :returns: a vector of len(theta1) containing the reflectivity
             value corresponding to each angle.
    """
    theta1 = np.radians(theta1)

    ip1 = vp1 * rho1
    ip2 = vp2 * rho2
    rp0 = (ip2 - ip1) / (ip2 + ip1)

    pr2, pr1 = pr(vp2, vs2), pr(vp1, vs1)
    pravg = (pr2 + pr1) / 2.
    pr = (pr2 - pr1) / (1 - pravg)**2.

    term1 = rp0 * np.cos(theta1)**2.
    term2 = pr * np.sin(theta1)**2.

    if terms:
        fields = ['term1', 'term2']
        Hilterman = namedtuple('Hilterman', fields)
        return Hilterman(term1, term2)
    else:
        return (term1 + term2)


def blangy(vp1, vs1, rho1, vp2, vs2, rho2,
           d1=0, e1=0, d2=0, e2=0,
           theta1=0, terms=False):
    """Implements the Blangy equation with the same interface as the other
    reflectivity equations. Wraps bruges.anisotropy.blangy().

    Note that the anisotropic parameters come after the other rock properties,
    and all default to zero.

    :param vp1: The p-wave velocity of the upper medium.
    :param vs1: The s-wave velocity of the upper medium.
    :param rho1: The density of the upper medium.
    :param vp2: The p-wave velocity of the lower medium.
    :param vs2: The s-wave velocity of the lower medium.
    :param rho2: The density of the lower medium.
    :param d1: Thomsen's delta for the upper medium.
    :param e1: Thomsen's epsilon for the upper medium.
    :param d2: Thomsen's delta for the upper medium.
    :param e2: Thomsen's epsilon for the upper medium.
    :param theta1: An array of incident angles to use for reflectivity
                   calculation [degrees].

    :returns: a vector of len(theta1) containing the reflectivity
             value corresponding to each angle.
    :param theta1: An array of incident angles to use for reflectivity
                   calculation [degrees].

    :returns: a vector of len(theta1) containing the reflectivity
             value corresponding to each angle.
    """
    _, anisotropic = anisotropy.blangy(vp1, vs1, rho1, d1, e1,  # UPPER
                                       vp2, vs2, rho2, d2, e2,  # LOWER
                                       theta1)
    return anisotropic


#-----------------the followings are from aadm's \geophyscal_note-master\aawedge.py---------------------
'''
===================
aawedge.py
===================

Functions to build and plot seismic wedges.

Created April 2015 by Alessandro Amato del Monte (alessandro.adm@gmail.com)

Heavily inspired by Matt Hall and Evan Bianco's blog posts and code:

http://nbviewer.ipython.org/github/agile-geoscience/notebooks/blob/master/To_make_a_wedge.ipynb
http://nbviewer.ipython.org/github/kwinkunks/notebooks/blob/master/Spectral_wedge.ipynb
http://nbviewer.ipython.org/github/kwinkunks/notebooks/blob/master/Faster_wedges.ipynb
http://nbviewer.ipython.org/github/kwinkunks/notebooks/blob/master/Variable_wedge.ipynb

Also see Wes Hamlyn's tutorial on Leading Edge "Thin Beds, tuning and AVO" (December 2014):

https://github.com/seg/tutorials/tree/master/1412_Tuning_and_AVO

HISTORY
2015-05-07 updated make_synth, now works also on 1D arrays.
2015-04-10 first public release.
'''

import numpy as np
import matplotlib.pyplot as plt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_wedge(n_traces,encasing_thickness,min_thickness,max_thickness,dz=0.1):
    '''
    Creates wedge-shaped model made of 3 units with variable thickness.

    INPUT
    n_traces
    encasing_thickness
    min_thickness
    max_thickness
    dz: vertical sample rate, by default 0.1 m

    OUTPUT
    wedge: 2D numpy array containing wedge-shaped model made of 3 units
    '''
    encasing_thickness *= (1./dz)
    min_thickness *= (1./dz)
    max_thickness *= (1./dz)
    deltaz=float(max_thickness-min_thickness)/float(n_traces)
    n_samples=max_thickness+encasing_thickness*2
    top_wedge=encasing_thickness
    wedge = np.zeros((n_samples, n_traces))
    wedge[0:encasing_thickness,:]=1
    wedge[encasing_thickness:,:]=3
    wedge[encasing_thickness:encasing_thickness+min_thickness,:]=2
    for i in range(n_traces):
        wedge[encasing_thickness+min_thickness:encasing_thickness+min_thickness+int(round(deltaz*i)),i]=2
    
    print (f"wedge minimum thickness: %.2f m" % (min_thickness*dz))
    print (f"wedge maximum thickness: %.2f m" % (max_thickness*dz))
    print (f"wedge vertical sampling: %.2f m" % (dz))
    print (f"wedge samples, traces: %dx%d" % (wedge.shape))
    return wedge

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def assign_ai(model, aiprop):
    '''
    Assigns acoustic impedance to a rock model created with make_wedge.

    INPUT
    model: 2D numpy array containing values from 1 to 3
    aiprop: np.array([[vp1,rho1],[vp2,rho2],[vp3,rho3]])

    OUTPUT
    model_ai: 2D numpy array containing acoustic impedances
    '''
    model_ai=np.zeros(model.shape)
    code = 1
    for x in aiprop:
        model_ai[model==code] = x[0]*x[1]
        code += 1
    return model_ai

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def assign_vel(model, aiprop):
    '''
    Assigns velocity to a rock model created with make_wedge,
    to be used for depth-time conversion.

    INPUT
    model: 2D numpy array containing values from 1 to 3
    aiprop: np.array([[vp1,rho1],[vp2,rho2],[vp3,rho3]])

    OUTPUT
    model_vel: 2D numpy array containing velocities
    '''
    model_vel=np.zeros(model.shape)
    code=1
    for x in aiprop:
        model_vel[model==code] = x[0]
        code += 1
    return model_vel

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def assign_el(model, elprop):
    '''
    Assigns elastic properties (Vp, Vs, rho) to a rock model created with make_wedge.

    INPUT
    model: 2D numpy array containing values from 1 to 3
    elprop: np.array([[vp1,rho1,vs1],[vp2,rho2,vs2],[vp3,rho3,vs3]])

    OUTPUT
    model_vp: 2D numpy array containing Vp
    model_vs: 2D numpy array containing Vs
    model_rho: 2D numpy array containing densities
    '''
    model_vp=np.zeros(model.shape)
    model_vs=np.zeros(model.shape)
    model_rho=np.zeros(model.shape)
    code = 1
    for i in elprop:
        model_vp[model==code]  = i[0]
        model_vs[model==code]  = i[2]
        model_rho[model==code] = i[1]
        code += 1
    return model_vp,model_vs,model_rho

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_rc(model_ai):
    '''
    Computes reflectivities of an acoustic model created with make_wedge + assign_ai.

    INPUT
    model: 2D numpy array containing acoustic impedances

    OUTPUT
    rc: 2D numpy array containing reflectivities
    '''
    upper = model_ai[:-1][:][:]
    lower = model_ai[1:][:][:]
    rc=(lower - upper) / (lower + upper)
    if model_ai.ndim==1:
        rc=np.concatenate((rc,[0]))
    else:
        n_traces=model_ai.shape[1]
        rc=np.concatenate((rc,np.zeros((1,n_traces))))  # add 1 row of zeros at the end
    return rc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_rc_elastic(model_vp,model_vs,model_rho,ang):
    '''
    Computes angle-dependent reflectivities of an elastic model created with make_wedge + assign_el.
    Uses Aki-Richards approximation.

    INPUT
    model_vp: 2D numpy array containing Vp values
    model_vs: 2D numpy array containing Vs values
    model_rho: 2D numpy array containing density values
    ang: list with near, mid, far angle, e.g. ang=[5,20,40]

    OUTPUT
    rc_near: 2D numpy array containing near-stack reflectivities
    rc_mid: 2D numpy array containing mid-stack reflectivities
    rc_far: 2D numpy array containing far-stack reflectivities
    '''
    [n_samples, n_traces] = model_vp.shape
    rc_near=np.zeros((n_samples,n_traces))
    rc_mid=np.zeros((n_samples,n_traces))
    rc_far=np.zeros((n_samples,n_traces))
    uvp  = model_vp[:-1][:][:]
    lvp  = model_vp[1:][:][:]
    uvs  = model_vs[:-1][:][:]
    lvs  = model_vs[1:][:][:]
    urho = model_rho[:-1][:][:]
    lrho = model_rho[1:][:][:]
    rc_near=akirichards(uvp,uvs,urho,lvp,lvs,lrho,ang[0])
    rc_mid=akirichards(uvp,uvs,urho,lvp,lvs,lrho,ang[1])
    rc_far=akirichards(uvp,uvs,urho,lvp,lvs,lrho,ang[2])
    rc_near=np.concatenate((rc_near,np.zeros((1,n_traces))))  # add 1 row of zeros at the end
    rc_mid=np.concatenate((rc_mid,np.zeros((1,n_traces))))
    rc_far=np.concatenate((rc_far,np.zeros((1,n_traces))))
    return rc_near, rc_mid, rc_far


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_synth(rc,wavelet):
    '''
    Convolves reflectivities with wavelet.

    INPUT
    rc: 2D numpy array containing reflectivities
    wavelet

    OUTPUT
    synth: 2D numpy array containing seismic data

    Works with 1D arrays now (2015-05-07).
    '''
    nt=np.size(wavelet)
    if rc.ndim>1:
        [n_samples, n_traces] = rc.shape
        synth = np.zeros((n_samples+nt-1, n_traces))
        for i in range(n_traces):
            synth[:,i] = np.convolve(rc[:,i], wavelet)
        synth = synth[np.ceil(len(wavelet))/2:-np.ceil(len(wavelet))/2, :]
        synth=np.concatenate((synth,np.zeros((1,n_traces))))
    else:
        n_samples = rc.size
        synth = np.zeros(n_samples+nt-1)
        synth = np.convolve(rc, wavelet)
        synth = synth[np.ceil(len(wavelet))/2:-np.ceil(len(wavelet))/2]
        synth=np.concatenate((synth,[0]))
    return synth

# def make_synth(rc,wavelet):
#     nt=np.size(wavelet)
#     [n_samples, n_traces] = rc.shape
#     synth = np.zeros((n_samples+nt-1, n_traces))
#     for i in range(n_traces):
#         synth[:,i] = np.convolve(rc[:,i], wavelet)
#     synth = synth[np.ceil(len(wavelet))/2:-np.ceil(len(wavelet))/2, :]
#     synth=np.concatenate((synth,np.zeros((1,n_traces))))
#     return synth
#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_synth_v2(rc,wavelet):
    '''
    Convolves reflectivities with wavelet.
    Alternative version using numpy apply_along_axis,
    slower than np.convolve with for loop.

    INPUT
    rc: 2D numpy array containing reflectivities
    wavelet

    OUTPUT
    synth: 2D numpy array containing seismic data
    '''
    nt=np.size(wavelet)
    [n_samples, n_traces] = rc.shape
    synth=np.zeros((n_samples+nt-1, n_traces))
    synth=np.apply_along_axis(lambda m: np.convolve(m,wavelet),axis=0,arr=rc)
    return synth

#~~~~~~~~~~~~~3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_synth_v3(rc,wavelet):
    '''
    Convolves reflectivities with wavelet.
    Alternative version using scipy.ndimage.filters.convolve1d,
    slower than np.convolve with for loop.

    INPUT
    rc: 2D numpy array containing reflectivities
    wavelet

    OUTPUT
    synth: 2D numpy array containing seismic data
    '''
    from scipy.ndimage.filters import convolve1d
    nt=np.size(wavelet)
    [n_samples, n_traces] = rc.shape
    synth=np.zeros((n_samples+nt-1, n_traces))
    synth=convolve1d(rc,wavelet,axis=0)
    return synth

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def forward_model(model,aiprop,wavelet,dz,dt):
    """
    Meta function to do everything from scratch (zero-offset model).
    """
    earth = assign_ai(model, aiprop)
    vels = assign_vel(model, aiprop)
    earth_time=agilegeo.avo.depth_to_time(earth,vels,dz,dt,twt=True)
    rc = make_rc(earth_time)
    return make_synth(rc,wavelet)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def forward_model_elastic(model,elprop,wavelet,ang,dz,dt):
    """
    Meta function to do everything from scratch (angle-dependent models).
    """
    model_vp,model_vs,model_rho = assign_el(model,elprop)
    model_vp_time=agilegeo.avo.depth_to_time(model_vp,model_vp,dz,dt,twt=True)
    model_vs_time=agilegeo.avo.depth_to_time(model_vs,model_vp,dz,dt,twt=True)
    model_rho_time=agilegeo.avo.depth_to_time(model_rho,model_vp,dz,dt,twt=True)

    rc_near, rc_mid, rc_far=make_rc_elastic(model_vp_time,model_vs_time,model_rho_time,ang)
    near = make_synth(rc_near,wavelet)
    mid = make_synth(rc_mid,wavelet)
    far = make_synth(rc_far,wavelet)
    return near,mid,far

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def forward_model_elastic_decay(model,elprop,wav_near,wav_mid,wav_far,dz,dt):
    """
    Meta function to do everything from scratch (angle-dependent models).
    Uses angle-dependent wavelet to simulate frequency decay with offset.
    """
    model_vp,model_vs,model_rho = assign_el(model,elprop)
    model_vp_time=agilegeo.avo.depth_to_time(model_vp,model_vp,dz,dt,twt=True)
    model_vs_time=agilegeo.avo.depth_to_time(model_vs,model_vp,dz,dt,twt=True)
    model_rho_time=agilegeo.avo.depth_to_time(model_rho,model_vp,dz,dt,twt=True)

    rc_near, rc_mid, rc_far=make_rc_elastic(model_vp_time,model_vs_time,model_rho_time,ang)
    near = make_synth(rc_near,wav_near)
    mid = make_synth(rc_mid,wav_mid)
    far = make_synth(rc_far,wav_far)
    return near,mid,far

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def extract_amp(data,elprop,encasing_thickness,min_thickness,max_thickness,dt,freq):
    '''
    Extracts top and bottom real/apparent amplitudes from wedge.

    INPUT
    data: synthetic wedge in twt
    elprop: np.array([[vp1,rho1,vs1],[vp2,rho2,vs2],[vp3,rho3,vs3]])
    encasing_thickness
    min_thickness
    max_thickness
    dt: twt vertical sample rate

    OUTPUT
    toptwt0,bottwt0: top, bottom horizon (REAL)
    topamp0,botamp0: top, bottom amplitude (REAL)
    toptwt1,bottwt1: top, bottom horizon (APPARENT)
    topamp1,botamp1: top, bottom amplitude (APPARENT)
    '''
    [ns,nt]=data.shape
    twt=np.arange(0,ns*dt,dt)

    Fd=freq*1.3
    b=1/Fd
    cerca=int((b/dt)/2)

    # if Ip_above<Ip_below then we have an INCREASE in Ip = positive RC = peak
    top_is_peak=elprop[0,0]*elprop[0,1]<elprop[1,0]*elprop[1,1]
    bot_is_peak=elprop[1,0]*elprop[1,1]<elprop[2,0]*elprop[2,1]

    layer_1_twt=float(encasing_thickness)/elprop[0,0]*2
    incr=(max_thickness-min_thickness)/float(nt)

    toptwt0=np.zeros(nt)+layer_1_twt
    bottwt0=np.zeros(nt)+layer_1_twt+(min_thickness/elprop[1,0]*2)
    for i in range(nt):
        bottwt0[i]=bottwt0[i]+incr*i/elprop[1,0]*2

    # amplitude extraction at top,bottom REAL
    topamp0=np.zeros(nt)
    botamp0=np.zeros(nt)

    for i,val in enumerate(toptwt0):
        dd=np.abs(twt-val).argmin()
        window=data[dd,i]
        if top_is_peak:
            topamp0[i]=window.max()
        else:
            topamp0[i]=window.min()

    for i,val in enumerate(bottwt0):
        dd=np.abs(twt-val).argmin()
        window=data[dd,i]
        if bot_is_peak:
            botamp0[i]=window.max()
        else:
            botamp0[i]=window.min()

    # amplitude extraction at top,bottom APPARENT
    toptwt1=np.copy(toptwt0)
    bottwt1=np.copy(bottwt0)
    topamp1=np.zeros(nt)
    botamp1=np.zeros(nt)

    for i,val in enumerate(toptwt0):
        dd=np.abs(twt-val).argmin() # sample corresponding to horizon pick
        window=data[dd-cerca:dd+cerca,i] # amplitudes within a window centered on horizon pick and spanning -/+ samples (`cerca`)
        if np.any(window):
            if top_is_peak:
                toptwt1[i]=twt[np.abs(data[:,i]-window.max()).argmin()]
                topamp1[i]=window.max()
            else:
                toptwt1[i]=twt[np.abs(data[:,i]-window.min()).argmin()]
                topamp1[i]=window.min()
        else:
            toptwt1[i]=np.NaN
            topamp1[i]=np.NaN

    for i,val in enumerate(bottwt0):
        dd=np.abs(twt-val).argmin()
        window=data[dd-cerca:dd+cerca,i]
        if np.any(window):
            if bot_is_peak:
                bottwt1[i]=twt[np.abs(data[:,i]-window.max()).argmin()]
                botamp1[i]=window.max()
            else:
                bottwt1[i]=twt[np.abs(data[:,i]-window.min()).argmin()]
                botamp1[i]=window.min()
        else:
            bottwt1[i]=np.NaN
            botamp1[i]=np.NaN

    return toptwt0,bottwt0,topamp0,botamp0,toptwt1,bottwt1,topamp1,botamp1


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def extract_peakfreqs(data,min_thickness,max_thickness,dt):
    '''
    Extracts peak frequencies from wedge.

    INPUT
    data: synthetic wedge in twt
    min_thickness
    max_thickness
    dt: twt vertical sample rate

    OUTPUT
    aft: array with peak amplitude (A) at row 0, peak frequency (F) at row 1, thickness (T) at row 2
    spectra: array with amplitude spectra for all traces
    '''

    import aaplot
    from scipy.signal import argrelmax
    [ns,nt]=data.shape

    amp0,ff0=aaplot.ampspec(data[:,0],dt)
    spectra=np.zeros((amp0.size,nt))
    aft=np.zeros((3,nt)) # row 0: peak Amplitudes, row 1: peak Frequencies, row 2: Thickness
    for i in range(nt):
        amp,ff=aaplot.ampspec(data[:,i],dt)
        spectra[:,i]=amp
        peak_freq_list=ff[argrelmax(amp)]
        peak_amp_list=amp[argrelmax(amp)]
        if peak_freq_list.size==0:
            aft[0,i]=np.NaN
            aft[1,i]=np.NaN
        else:
            uu=peak_amp_list==np.max(peak_amp_list)
            peak_amp=peak_amp_list[uu]
            peak_freq=peak_freq_list[uu]
            aft[0,i]=peak_amp
            aft[1,i]=peak_freq
        incr=(max_thickness-min_thickness)/float(nt)
        aft[2,i]=i*incr+min_thickness
        # print peak_freq_list, peak_amp_list
        # print 'traccia %d, peak freq=%.2f, spessore=%.2f' % (i, peak_freq, ss[2,i])
    return aft, spectra

#---------------------------------------------------------------------------------------------------------------
"""
from aadm's notes:
5 functions to implement some of the easier (to code) rock physics models (RPM):

1)critical porosity (Nur et al., 1991, 1995);
2)Hertz-Mindlin, at the basis of soft and stiff sand models;
3)soft sand model (Dvorkin and Nur, 1996)
4)stiff sand model
5)cemented sand model (or contact cement model; Dvorkin and Nur, 1996)
6)Vernik and Kachanov's models:
    a) consolidated sand
    b) soft sand 1
    c) soft sand 2
    d) sandstone diagenesis
    e) shale

"""
import numpy as np
import matplotlib.pyplot as plt
def critpor(K0, G0, phi, phic=0.4):
    '''
    Critical porosity, Nur et al. (1991, 1995)
    written by aadm (2015) from Rock Physics Handbook, p.353

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    '''
    K_DRY  = K0 * (1-phi/phic)
    G_DRY  = G0 * (1-phi/phic)
    return K_DRY, G_DRY

def hertzmindlin(K0, G0, phic=0.4, Cn=8.6, sigma=10, f=1):
    '''
    Hertz-Mindlin model
    written by aadm (2015) from Rock Physics Handbook, p.246

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    Cn: coordination nnumber (default 8.6)
    sigma: effective stress in MPa (default 10)
    f: shear modulus correction factor
       1=dry pack with perfect adhesion
       0=dry frictionless pack
    '''
    sigma /= 1e3 # converts pressure in same units as solid moduli (GPa)
    PR0=(3*K0-2*G0)/(6*K0+2*G0) # poisson's ratio of mineral mixture
    K_HM = (sigma*(Cn**2*(1-phic)**2*G0**2) / (18*np.pi**2*(1-PR0)**2))**(1/3)
    G_HM = ((2+3*f-PR0*(1+3*f))/(5*(2-PR0))) * ((sigma*(3*Cn**2*(1-phic)**2*G0**2)/(2*np.pi**2*(1-PR0)**2)))**(1/3)
    return K_HM, G_HM

def softsand(K0, G0, phi, phic=0.4, Cn=8.6, sigma=10, f=1):
    '''
    Soft-sand (uncemented) model
    written by aadm (2015) from Rock Physics Handbook, p.258

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    Cn: coordination nnumber (default 8.6)
    sigma: effective stress in MPa (default 10)
    f: shear modulus correction factor
       1=dry pack with perfect adhesion
       0=dry frictionless pack
    '''
    K_HM, G_HM = hertzmindlin(K0, G0, phic, Cn, sigma, f)
    K_DRY =-4/3*G_HM + (((phi/phic)/(K_HM+4/3*G_HM)) + ((1-phi/phic)/(K0+4/3*G_HM)))**-1
    tmp = G_HM/6*((9*K_HM+8*G_HM) / (K_HM+2*G_HM))
    G_DRY = -tmp + ((phi/phic)/(G_HM+tmp) + ((1-phi/phic)/(G0+tmp)))**-1
    return K_DRY, G_DRY

def stiffsand(K0, G0, phi, phic=0.4, Cn=8.6, sigma=10, f=1):
    '''
    Stiff-sand model
    written by aadm (2015) from Rock Physics Handbook, p.260

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    Cn: coordination nnumber (default 8.6)
    sigma: effective stress in MPa (default 10)
    f: shear modulus correction factor
       1=dry pack with perfect adhesion
       0=dry frictionless pack
    '''
    K_HM, G_HM = hertzmindlin(K0, G0, phic, Cn, sigma, f)
    K_DRY = -4/3*G0 + (((phi/phic)/(K_HM+4/3*G0)) + ((1-phi/phic)/(K0+4/3*G0)))**-1
    tmp = G0/6*((9*K0+8*G0) / (K0+2*G0))
    G_DRY = -tmp + ((phi/phic)/(G_HM+tmp) + ((1-phi/phic)/(G0+tmp)))**-1
    return K_DRY, G_DRY

def contactcement(K0, G0, phi, phic=0.4, Cn=8.6, Kc=37, Gc=45, scheme=2):
    '''
    Contact cement (cemented sand) model, Dvorkin-Nur (1996)
    written by aadm (2015) from Rock Physics Handbook, p.255

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    Cn: coordination nnumber (default 8.6)
    Kc, Gc: cement bulk & shear modulus in GPa
            (default 37, 45 i.e. quartz)
    scheme: 1=cement deposited at grain contacts
            2=uniform layer around grains (default)
    '''
    PR0=(3*K0-2*G0)/(6*K0+2*G0)
    PRc = (3*Kc-2*Gc)/(6*Kc+2*Gc)
    if scheme == 1: # scheme 1: cement deposited at grain contacts
        alpha = ((phic-phi)/(3*Cn*(1-phic))) ** (1/4)
    else: # scheme 2: cement evenly deposited on grain surface
        alpha = ((2*(phic-phi))/(3*(1-phic)))**(1/2)
    LambdaN = (2*Gc*(1-PR0)*(1-PRc)) / (np.pi*G0*(1-2*PRc))
    N1 = -0.024153*LambdaN**-1.3646
    N2 = 0.20405*LambdaN**-0.89008
    N3 = 0.00024649*LambdaN**-1.9864
    Sn = N1*alpha**2 + N2*alpha + N3
    LambdaT = Gc/(np.pi*G0)
    T1 = -10**-2*(2.26*PR0**2+2.07*PR0+2.3)*LambdaT**(0.079*PR0**2+0.1754*PR0-1.342)
    T2 = (0.0573*PR0**2+0.0937*PR0+0.202)*LambdaT**(0.0274*PR0**2+0.0529*PR0-0.8765)
    T3 = 10**-4*(9.654*PR0**2+4.945*PR0+3.1)*LambdaT**(0.01867*PR0**2+0.4011*PR0-1.8186)
    St = T1*alpha**2 + T2*alpha + T3
    K_DRY = 1/6*Cn*(1-phic)*(Kc+(4/3)*Gc)*Sn
    G_DRY = 3/5*K_DRY+3/20*Cn*(1-phic)*Gc*St
    return K_DRY, G_DRY

def vernik_csm(K0, G0, phi, sigma, b=10):
    '''
    vernik_csm (C) aadm 2017
    Vernik & Kachanov Consolidated Sand Model.

    reference:
    Vernik & Kachanov (2010), Modeling elastic properties of siliciclastic rocks, Geophysics v.75 n.6

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    sigma: effective stress in MPa
    b: slope parameter in pore shape empirical equation (default 10, range 8-12)

    OUTPUT
    K_DRY, G_DRY: dry rock bulk & shear modulus in GPa
    '''
    # empirical pore shape factor
    p = 3.6+b*phi
    q = p # true if phi>0.03
    PSF = phi/(1-phi) # PSF = pore shape factor multiplier

    # matrix properties: assuming arenites w/ properties K=35.6 GPa, G=33 GPa, poisson's ratio nu_m = 0.146
    nu_m = 0.146
    Avm = (16*(1-nu_m**2) )/( 9*(1-2*nu_m))      # nu_m=0.146 --> Avm=2.46
    Bvm = (32*(1-nu_m)*(5-nu_m) )/( 45*(2-nu_m)) # nu_m=0.146 --> Bvm=1.59

    # crack density: inversely correlated to effective stress
    eta0 = 0.3+1.6*phi # crack density at zero stress
    d = 0.07 # compaction coefficient
    d = 0.02+0.003*sigma
    CD = (eta0 * np.exp(-d * sigma))/(1-phi)  # sigma=stress in MPa

    # note: the presence at denominator of the factor (1-phi) in PSF and CD is needed
    # to account for the interaction effects, i.e. the presence of pores raises the average stress
    # in the matrix increasing compliance contributions of pores and cracks
    # this correction is referred to as Mori-Tanaka's scheme.
    # in this way, the original model which is a NIA (non-interaction model)
    # is extended and becomes effectively a model which does take into account interactions.
    Kdry = K0*(1+p*PSF+Avm*CD)**-1
    Gdry = G0*(1+q*PSF+Bvm*CD)**-1
    return Kdry, Gdry


def vernik_ssm1(K0, G0, phi, sigma, phi_c=0.36, phi_con=0.26, b=10, n=2.00, m=2.05):
    '''
    vernik_ssm1 (C) aadm 2017
    Vernik & Kachanov Soft Sand Model 1.
    Only applicable for sands with porosity between phi_c and phi_con.

    reference:
    Vernik & Kachanov (2010), Modeling elastic properties of siliciclastic rocks, Geophysics v.75 n.6

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    sigma: effective stress in MPa
    phi_c: critical porosity (default 0.36, range 0.30-0.42)
    phi_con: consolidation porosity (default 0.26, range 0.22-0.30)
    b: slope parameter in pore shape empirical equation (default 10, range 8-12)
    n, m: empirical factors (default 2.00, 2.05)

    OUTPUT
    K_DRY, G_DRY: dry rock bulk & shear modulus in GPa
    '''
    if isinstance(phi, np.ndarray):
        phi_edit = phi.copy()
        phi_edit[(phi_edit<phi_con) | (phi_edit>phi_c)]=np.nan
    else:
        phi_edit = np.array(phi)
        if (phi_edit<phi_con) | (phi_edit>phi_c):
            return np.nan, np.nan
    M0 = K0+4/3*G0
    K_con, G_con = vernik_csm(K0,G0,phi_con,sigma, b)
    M_con = K_con+4/3*G_con
    T = (1-(phi_edit-phi_con)/(phi_c-phi_con))
    Mdry = M_con*T**n
    Gdry = G_con*T**m
    Kdry = Mdry-4/3*Gdry
    return Kdry, Gdry


def vernik_ssm2(K0, G0, phi, p=20, q=20):
    '''
    vernik_ssm2 (C) aadm 2017
    Vernik & Kachanov Soft Sand Model 2.

    reference:
    Vernik & Kachanov (2010), Modeling elastic properties of siliciclastic rocks, Geophysics v.75 n.6

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    p, q: pore shape factor for K and G (default p=q=20,range: 10-45)

    OUTPUT
    K_DRY, G_DRY: dry rock bulk & shear modulus in GPa
    '''
    M0 = K0+4/3*G0
    Mdry = M0*(1+p*(phi/(1-phi)))**-1
    Gdry = G0*(1+q*(phi/(1-phi)))**-1
    Kdry = Mdry-4/3*Gdry
    return Kdry, Gdry

def vernik_sdm(K0, G0, phi, sigma, phi_c=0.36, phi_con=0.26, b=10, n=2.00, m=2.05):
    '''
    vernik_sdm (C) aadm 2017
    Vernik & Kachanov Sandstone Diagenesis Model.
    Combination of CSM and SSM1 (for porosity>phi_con) models.

    reference:
    Vernik & Kachanov (2010), Modeling elastic properties of siliciclastic rocks, Geophysics v.75 n.6

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa

    phi: porosity
    sigma: effective stress in MPa
    phi_c: critical porosity (default 0.36, range 0.30-0.42)
    phi_con: consolidation porosity (default 0.26, range 0.22-0.30)
    b: slope parameter in pore shape empirical equation (default 10, range 8-12)
    n, m: empirical factors (default 2.00, 2.05)

    OUTPUT
    K_DRY, G_DRY: dry rock bulk & shear modulus in GPa
    '''
    Kdry, Gdry = vernik_csm(K0, G0, phi, sigma, b)
    Kdry_soft, Gdry_soft = vernik_ssm1(K0, G0, phi, sigma, phi_c, phi_con, b, n, m)
    if isinstance(phi, np.ndarray):
        uu=phi>=phi_con
        Kdry[uu] = Kdry_soft[uu]
        Gdry[uu] = Gdry_soft[uu]
        return Kdry, Gdry
    else:
        if phi<=phi_con:
            return Kdry,Gdry
        else:
            return Kdry_soft, Gdry_soft

def vernik_shale(vclay, phi, rhom=2.73, rhob=1, Mqz=96, c33_clay=33.4, A=0.00284):
    '''
    vernik_shale (C) aadm 2017
    Vernik & Kachanov Shale Model.

    Shale matrix density (rhom) averages 2.73 +/- 0.03 g/cc at porosities below 0.25.
    It gradually varies with compaction and smectite-to-illite transition.
    A more accurate estimate can be calculated with this equation:
    rhom = 2.76+0.001*((rho-2)-230*np.exp(-4*(rho-2)))

    reference:
    Vernik & Kachanov (2010), Modeling elastic properties of siliciclastic rocks, Geophysics v.75 n.6

    INPUT
    vclay: dry clay content volume fraction
    phi: porosity (maximum 0.40)
    rhom: shale matrix density (g/cc, default 2.73)
    rhob: brine density (g/cc, default 1)
    Mqz: P-wave elastic modulus of remaining minerals (GPa, default 96)
    c33_clay: anisotropic clay constant (GPa, default 33.4)
    A: empirical coefficient for Vs (default .00284 for illite/smectite/chlorite,
       can be raised up to .006 for kaolinite-rich clays)

    OUTPUT
    vp, vs, density: P- and S-wave velocities in m/s, density in g/cc
    '''
    rho_matrix = 2.65*(1-vclay)+rhom*vclay
    k = 5.2-1.3*vclay
    B, C = 0.287, 0.79
    c33_min = (vclay/c33_clay + (1-vclay)/Mqz)**-1
    c33 = c33_min*(1-phi)**k
    vp = np.sqrt(c33/(rhom*(1-phi)+rhob*phi))
    vs = np.sqrt(A*vp**4 + B*vp**2 - C)
    rho = rho_matrix*(1-phi)+rhob*phi
    return vp*1e3,vs*1e3, rho

def vels(K_DRY,G_DRY,K0,D0,Kf,Df,phi):
    '''
    Calculates velocities and densities of saturated rock via Gassmann equation, (C) aadm 2015

    INPUT
    K_DRY,G_DRY: dry rock bulk & shear modulus in GPa
    K0, D0: mineral bulk modulus and density in GPa
    Kf, Df: fluid bulk modulus and density in GPa
    phi: porosity
    '''
    rho  = D0*(1-phi)+Df*phi
    K = K_DRY + (1-K_DRY/K0)**2 / ( (phi/Kf) + ((1-phi)/K0) - (K_DRY/K0**2) )
    vp   = np.sqrt((K+4./3*G_DRY)/rho)*1e3
    vs   = np.sqrt(G_DRY/rho)*1e3
    return vp, vs, rho, K

def vrh(volumes,k,mu):
    '''
    Calculates Voigt-Reuss-Hill bounds, (C) aadm 2015

    INPUT
    volumes: array with volumetric fractions
    k: array with bulk modulus
    mu: array with shear modulus

    OUTPUT
    k_u, k_l: upper (Voigt) and lower (Reuss) average of k
    mu_u, mu_l: upper (Voigt) and lower (Reuss) average of mu
    k0, mu0: Hill average of k and mu
    '''
    f=np.array(volumes).T
    k=np.resize(np.array(k),np.shape(f))
    mu=np.resize(np.array(mu),np.shape(f))
    ax=0 if f.ndim==1 else 1
    k_u = np.sum(f*k,axis=ax)
    k_l = 1./np.sum(f/k,axis=ax)
    mu_u = np.sum(f*mu,axis=ax)
    mu_l = 1./np.sum(f/mu,axis=ax)
    k0 = (k_u+k_l)/2.
    mu0 = (mu_u+mu_l)/2.
    return k_u, k_l, mu_u, mu_l, k0, mu0
