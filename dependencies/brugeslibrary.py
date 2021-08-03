import functools
import inspect
import warnings
import scipy.signal
import scipy.ndimage
import numpy as np
from collections import namedtuple
from scipy.signal import hilbert
from scipy.signal import chirp
from functools import wraps
import scipy.ndimage as sn
import scipy.ndimage.morphology as morph
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.fftpack import fft
from scipy.signal import get_window
#================================================================================================
# UTILITY FUNCTIONS
#================================================================================================
"""
Utility functions.

:copyright: 2021 Agile Scientific
:license: Apache 2.0
"""
def deprecated(instructions):
    """
    Flags a method as deprecated. This decorator can be used to mark functions
    as deprecated. It will result in a warning being emitted when the function
    is used.
    Args:
        instructions (str): A human-friendly string of instructions, such
            as: 'Please migrate to add_proxy() ASAP.'
    Returns:
        The decorated function.
    """
    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = 'Call to deprecated function {}. {}'.format(
                func.__name__,
                instructions)

            frame = inspect.currentframe().f_back

            warnings.warn_explicit(message,
                                   category=DeprecationWarning,
                                   filename=inspect.getfile(frame.f_code),
                                   lineno=frame.f_lineno)

            return func(*args, **kwargs)

        return wrapper

    return decorator


greek = {
    'Alpha': 'Α',
    'Beta': 'Β',
    'Gamma': 'Γ',
    'Delta': 'Δ',
    'Epsilon': 'Ε',
    'Zeta': 'Ζ',
    'Eta': 'Η',
    'Kappa': 'Κ',
    'Lambda': 'Λ',
    'Mu': 'Μ',
    'Nu': 'Ν',
    'Phi': 'Φ',
    'Pi': 'Π',
    'Rho': 'Ρ',
    'Sigma': 'Σ',
    'Tau': 'Τ',
    'Upsilon': 'Υ',
    'Theta': 'Θ',
    'Chi': 'Χ',
    'Psi': 'Ψ',
    'Omega': 'Ω',
    'alpha': 'α',
    'beta': 'β',
    'gamma': 'γ',
    'delta': 'δ',
    'epsilon': 'ε',
    'zeta': 'ζ',
    'eta': 'η',
    'theta': 'θ',
    'kappa': 'κ',
    'lambda': 'λ',
    'mu': 'μ',
    'nu': 'ν',
    'pi': 'π',
    'rho': 'ρ',
    'sigma': 'σ',
    'tau': 'τ',
    'upsilon': 'υ',
    'phi': 'φ',
    'chi': 'χ',
    'psi': 'ψ',
    'omega': 'ω',
}


def rms(a, axis=None):
    """
    Calculates the RMS of an array.

    Args:
        a (ndarray). A sequence of numbers to apply the RMS to.
        axis (int). The axis along which to compute. If not given or None,
            the RMS for the whole array is computed.

    Returns:
        ndarray: The RMS of the array along the desired axis or axes.
    """
    a = np.array(a)
    if axis is None:
        div = a.size
    else:
        div = a.shape[axis]
    ms = np.sum(a**2.0, axis=axis) / div
    return np.sqrt(ms)


def moving_average(a, length, mode='same'):
    """
    Computes the mean in a moving window using convolution. For an alternative,
    as well as other kinds of average (median, mode, etc.), see bruges.filters.

    Example:
        >>> test = np.array([1,1,9,9,9,9,9,2,3,9,2,2,np.nan,1,1,1,1])
        >>> moving_average(test, 5, mode='same')
        array([ 2.2,  4. ,  5.8,  7.4,  9. ,  7.6,  6.4,  6.4,  5. ,  3.6,  nan,
                nan,  nan,  nan,  nan,  0.8,  0.6])
    """
    padded = np.pad(a, int(length/2), mode='edge')
    boxcar = np.ones(int(length))/length
    smooth = np.convolve(padded, boxcar, mode='same')
    return smooth[int(length/2):-int(length/2)]


@deprecated("Use bruges.filters() for moving linear and nonlinear statistics")
def moving_avg_conv(a, length, mode='same'):
    """
    Moving average via convolution. Keeping it for now for compatibility.
    """
    boxcar = np.ones(length)/length
    return np.convolve(a, boxcar, mode=mode)


@deprecated("Use bruges.filters() for moving linear and nonlinear statistics")
def moving_avg_fft(a, length, mode='same'):
    """
    Moving average via FFT convolution. Keeping it for now for compatibility.

    """
    boxcar = np.ones(length)/length
    return scipy.signal.fftconvolve(a, boxcar, mode=mode)


def normalize(a, new_min=0.0, new_max=1.0):
    """
    Normalize an array to [0,1] or to arbitrary new min and max.

    Args:
        a (ndarray): An array.
        new_min (float): The new min to scale to, default 0.
        new_max (float): The new max to scale to, default 1.

    Returns:
        ndarray. The normalized array.
    """
    a = np.array(a, dtype=np.float)
    n = (a - np.nanmin(a)) / np.nanmax(a - np.nanmin(a))
    return n * (new_max - new_min) + new_min


def nearest(a, num):
    """
    Finds the array's nearest value to a given num.

    Args:
        a (ndarray): An array.
        num (float): The value to find the nearest to.

    Returns:
        float. The normalized array.
    """
    a = np.array(a, dtype=float)
    return a.flat[np.abs(a - num).argmin()]


def next_pow2(num):
    """
    Calculates the next nearest power of 2 to the input. Uses
      2**ceil( log2( num ) ).

    Args:
        num (number): The number to round to the next power if two.

    Returns:
        number. The next power of 2 closest to num.
    """

    return int(2**np.ceil(np.log2(num)))


def top_and_tail(*arrays):
    """
    Top and tail all arrays to the non-NaN extent of the first array.

    E.g. crop the NaNs from the top and tail of a well log.

    Args:
        arrays (list): A list of arrays to treat.

    Returns:
        list: A list of treated arrays.
    """
    if len(arrays) > 1:
        for arr in arrays[1:]:
            assert len(arr) == len(arrays[0])
    nans = np.where(~np.isnan(arrays[0]))[0]
    first, last = nans[0], nans[-1]
    return [array[first:last+1] for array in arrays]


def extrapolate(a):
    """
    Extrapolate up and down an array from the first and last non-NaN samples.

    E.g. Continue the first and last non-NaN values of a log up and down.

    Args:
        a (ndarray): The array to treat.

    Returns:
        ndarray: The treated array.
    """
    a = np.array(a)
    nans = np.where(~np.isnan(a))[0]
    first, last = nans[0], nans[-1]
    a[:first] = a[first]
    a[last + 1:] = a[last]
    return a


def error_flag(pred, actual, dev=1.0, method=1):
    """
    Calculate the difference between a predicted and an actual curve
    and return a log flagging large differences based on a user-defined
    distance (in standard deviation units) from the mean difference.

    Args:
        predicted (ndarray): predicted log.
        actual (ndarray):  original log.
        dev (float): standard deviations to use, default 1
        error calcluation method (int): default 1
            1: difference between curves larger than mean difference plus dev
            2: curve slopes have opposite sign. Will require depth log for
               .diff method
            3: curve slopes of opposite sign OR difference larger than mean
               plus dev

    Returns:
        flag (ndarray) =  error flag curve

    Author:
        Matteo Niccoli, 2018
    """

    flag = np.zeros(len(pred))
    err = np.abs(pred-actual)
    err_mean = np.mean(err)
    err_std = np.std(err)

    if method == 1:
        flag[np.where(err > (err_mean + (dev * err_std)))] = 1

    return flag


def apply_along_axis(func_1d, arr, kernel, **kwargs):
    """
    Apply 1D function across 2D slice as efficiently as possible.

    Although `np.apply_along_axis` seems to do well enough, map usually
    seems to end up beig a bit faster.

    Args:
        func_1d (function): the 1D function to apply, e.g. np.convolve. Should
            take 2 or more arguments: the

    Example
    >>> apply_along_axes(np.convolve, reflectivity_2d, wavelet, mode='same') 
    """
    mapobj = map(lambda tr: func_1d(tr, kernel, **kwargs), arr)
    return np.array(list(mapobj))


def sigmoid(start, stop, num):
    """
    Nonlinear space following a logistic function.

    The function is asymptotic; the parameters used in the sigmoid
    gets within 0.5% of the target thickness in a wedge increasing
    from 0 to 2x the original thickness.
    """
    x = np.linspace(-5.293305, 5.293305, num)
    return start + (stop-start) / (1 + np.exp(-x))


def root(start, stop, num):
    """
    Nonlinear space following a sqrt function.
    """
    x = np.linspace(0, 1, num)
    y = np.sqrt(x)
    return min(start, stop) + abs(stop-start) * y


def power(start, stop, num):
    """
    Nonlinear space following a power function.
    """
    x = np.linspace(0, 8, num)
    y = 1 - 2**-x
    return min(start, stop) + abs(stop-start) * y


#================================================================================================
# ROCK PHYSICS
#================================================================================================
# ANISOTROPY
#=========================================================
"""
Anisotropy effects.

Backus anisotropy is from thin layers.

Hudson anisotropy is from crack defects.

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
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
    lam = moduli.lam(vp, vs, rho)
    mu = moduli.mu(vp, vs, rho)

    # Compute the layer parameters from Liner (2014) equation 2:
    a = rho * np.power(vp, 2.0)  # Acoustic impedance

    # Compute the Backus parameters from Liner (2014) equation 4:
    A1 = 4 * moving_average(mu*(lam+mu)/a, lb/dz, mode='same')
    A = A1 + np.power(moving_average(lam/a, lb/dz, mode='same'), 2.0)\
        / moving_average(1.0/a, lb/dz, mode='same')
    C = 1.0 / moving_average(1.0/a, lb/dz, mode='same')
    F = moving_average(lam/a, lb/dz, mode='same')\
        / moving_average(1.0/a, lb/dz, mode='same')
    L = 1.0 / moving_average(1.0/mu, lb/dz, mode='same')
    M = moving_average(mu, lb/dz, mode='same')

    BackusResult = namedtuple('BackusResult', ['A', 'C', 'F', 'L', 'M'])
    return BackusResult(A, C, F, L, M)

#
def vectorize(func):
    """
    Decorator to make sure the inputs are arrays. We also add a dimension
    to theta to make the functions work in an 'outer product' way.

    Takes a reflectivity function requiring Vp, Vs, and RHOB for 2 rocks
    (upper and lower), plus incidence angle theta, plus kwargs. Returns
    that function with the arguments transformed to ndarrays.
    """
    @wraps(func)
    def wrapper(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, **kwargs):
        vp = np.asanyarray(vp, dtype=float)
        vs = np.asanyarray(vs, dtype=float) + 1e-12  # Prevent singular matrix.
        rho = np.asanyarray(rho, dtype=float)
        lb = np.asanyarray(lb, dtype=float).reshape((-1, 1))
        dz = np.asanyarray(dz, dtype=float)
        return func(vp, vs, rho, lb, dz)
    return wrapper


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

    Args:
        vp (ndarray): P-wave interval velocity.
        vs (ndarray): S-wave interval velocity.
        rho (ndarray): Bulk density.
        lb (float): The Backus averaging length in m.
        dz (float): The depth sample interval in m.

    Returns:
        namedtuple: Qp and Qs.
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

    Args:
        vp (ndarray): P-wave interval velocity.
        vs (ndarray): S-wave interval velocity.
        rho (ndarray): Bulk density.
        lb (float): The Backus averaging length in m.
        dz (float): The depth sample interval in m.

    Returns:
        namedtuple: delta, epsilon and gamma.

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

    Args:
        porosity (float): Fractional porosity, phi.
        aspect (float): Aspect ratio, alpha.
        mu (float): Shear modulus, sometimes called G.
        lam (float): Lame's first parameter, lambda.
        pmod (float): Compressional modulus, M.

    Returns:
        float: Q_p
        float: Q_s
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
    Dvorkin et al. (2014), Eq 15.44 (aligned) and 15.48 (not aligned). You must
    provide one of the following: `pr`, or `vp` and `vs`, or `mu` and `pmod`.

    Args:
        mu (float): Shear modulus, sometimes called G.
        pmod (float): Compressional modulus, M.
        pr (float): Poisson's ratio, somtimes called v.
        vp (ndarray): P-wave interval velocity.
        vs (ndarray): S-wave interval velocity.
        aligned (bool): Either treats cracks as alligned (Default, True)
                        or assumes defects are randomly oriented (False)

    Returns:
        float: 2Q_s^-1
    """
    if pr is not None:
        x = (2 - 2*pr) / (1 - 2*pr)
    elif (vp is not None) and (vs is not None):
        x = vp**2 / vs**2
    elif (mu is not None) and (pmod is not None):
        x = pmod / mu
    else:
        raise TypeError("You must provide pr or (vp and vs) or (mu and pmod)")

    if aligned:
        return 0.25 * (x - 2)**2 * (3*x - 2) / (x**2 - x)
    else:
        a = 2*x / (3*x - 2)
        b = x / 3*(x - 1)
        return 1.25 * ((x - 2)**2 / (x - 1)) / (a + b)

#=========================================================
# BOUNDS
#=========================================================
"""
Bounds on effective elastic modulus.
:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
def voigt_bound(f, m):
    """
    The upper bound on the effective elastic modulus, mv of a
    mixture of N material phases. This is defined at the arithmetic
    average of the constituents.

    Args:
        f: list or array of N volume fractions (must sum to 1 or 100).
        m: elastic modulus of N constituents (list or array).

     Returns:
        mv: Voigt upper bound.

    """
    f = np.array(f).astype(float)

    if float(sum(f)) == 100.0:
        # fractions have been given in percent: scale to 1.
        f /= 100.0

    m = np.array(m)
    mv = np.sum(f * m)

    return mv


def reuss_bound(f, m):
    """
    The lower bound on the effective elastic modulus of a
    mixture of N material phases. This is defined at the harmonic
    average of the constituents. Same as Wood's equation for homogeneous mixed fluids.

    Args:
        f: list or array of N volume fractions (must sum to 1 or 100).
        m: elastic modulus of N constituents (list or array).

     Returns:
        mr: Reuss lower bound.
    """
    f = np.array(f).astype(float)

    if float(sum(f)) == 100.0:
        # fractions have been given in percent: scale to 1.
        f /= 100.0

    m = np.array(m)
    mr = 1.0 / np.sum(f / m)

    return mr


def hill_average(f, m):
    """
    The Hill average effective elastic modulus, mh of a
    mixture of N material phases. This is defined as the simple
    average of the Reuss (lower) and Voigt (upper) bounds.

    Args:
        f: list or array of N volume fractions (must sum to 1 or 100).
        m: elastic modulus of N constituents (list or array).

     Returns:
        mh: Hill average.
    """
    mv = voigt_bound(f, m)
    mr = reuss_bound(f, m)
    mh = (mv + mr) / 2.0

    return mh


def hashin_shtrikman(f, k, mu, modulus='bulk'):
    """
    Hashin-Shtrikman bounds for a mixture of two constituents.
    The best bounds for an isotropic elastic mixture, which give
    the narrowest possible range of elastic modulus without
    specifying anything about the geometries of the constituents.

    Args:
        f: list or array of volume fractions (must sum to 1.00 or 100%).
        k: bulk modulus of constituents (list or array).
        mu: shear modulus of constituents (list or array).
        modulus: A string specifying whether to return either the
            'bulk' or 'shear' HS bound.

    Returns:
        namedtuple: The Hashin Shtrikman (lower, upper) bounds.

    :source: Berryman, J.G., 1993, Mixture theories for rock properties
             Mavko, G., 1993, Rock Physics Formulas.

    : Written originally by Xingzhou 'Frank' Liu, in MATLAB
    : modified by Isao Takahashi, 4/27/99,
    : Translated into Python by Evan Bianco
    """
    def z_bulk(k, mu):
        return (4/3.) * mu

    def z_shear(k, mu):
        return mu * (9 * k + 8 * mu) / (k + 2 * mu) / 6

    def bound(f, k, z):
        return 1 / sum(f / (k + z)) - z

    f = np.array(f)
    if sum(f) == 100:
        f /= 100.0

    func = {'shear': z_shear,
            'bulk': z_bulk}

    k, mu = np.array(k), np.array(mu)
    z_min = func[modulus](np.amin(k), np.amin(mu))
    z_max = func[modulus](np.amax(k), np.amax(mu))

    fields = ['lower_bound', 'upper_bound']
    HashinShtrikman = namedtuple('HashinShtrikman', fields)
    return HashinShtrikman(bound(f, k, z_min), bound(f, k, z_max))

#=========================================================
# ELASTIC
#=========================================================
"""
Elastic impedance.

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
def elastic_impedance(vp, vs, rho, theta1,
                      k=None,
                      normalize=False,
                      constants=None,
                      use_sin=False,
                      rho_term=False):
    """
    Returns the elastic impedance as defined by Connolly, 1999; we are using
    the formulation reported in Whitcombe et al. (2001). Inputs should be
    shape m x 1, angles should be n x 1. The result will be m x n.

    Args:
        vp (ndarray): The P-wave velocity scalar or 1D array.
        vs (ndarray): The S-wave velocity scalar or 1D array.
        rho (ndarray): The bulk density scalar or 1D array.
        theta1 (ndarray): Incident angle(s) in degrees, scalar or array.
        k (float): A constant, see Connolly (1999). Default is None, which
            computes it from Vp and Vs.
        normalize (bool): if True, returns the normalized form of Whitcombe.
        constants (tuple): A sequence of 3 constants to use for normalization.
            If you don't provide this, then normalization just uses the means
            of the inputs. If these are scalars, the result will be the
            acoustic impedance (see Whitcombe et al., 2001).
        use_sin (bool): If True, use sin^2 for the first term, instead of
            tan^2 (see Connolly).
        rho_term (bool): If True, provide alternative form, with Vp factored
            out; use in place of density in generating synthetics in other
            software (see Connolly). In other words, the result can be
            multipled with Vp to get the elastic impedance.

    Returns:
        ndarray: The elastic impedance log at the specficied angle or angles.
    """
    theta1 = np.radians(theta1).reshape(-1, 1)
    if (np.nan_to_num(theta1) > np.pi/2.).any():
        raise ValueError("Incidence angle theta1 must be less than 90 deg.")

    alpha = np.asanyarray(vp, dtype=float)
    beta = np.asanyarray(vs, dtype=float)
    rho = np.asanyarray(rho, dtype=float)
    op = np.sin if use_sin else np.tan
    k = np.mean(beta**2.0 / alpha**2.0) if k is None else k

    a = 1 + op(theta1)**2.0
    b = -8 * k * np.sin(theta1)**2.0
    c = 1 - 4 * k * np.sin(theta1)**2.0

    ei = alpha**a * beta**b * rho**c

    if normalize:
        if constants is None:
            # Use the means; this will return acoustic impedance for scalars.
            alpha_0 = np.nanmean(vp)
            beta_0 = np.nanmean(vs)
            rho_0 = np.nanmean(rho)
        else:
            try:
                alpha_0, beta_0, rho_0 = constants
            except ValueError as e:
                raise ValueError("You must provide a sequence of 3 constants.")
        ei *= alpha_0**(1 - a) * beta_0**(-b) * rho_0**(1 - c)

    if rho_term:
        ei /= alpha

    if ei.size == 1:
        return np.asscalar(ei)
    else:
        return np.squeeze(ei.T)

#=========================================================
# FLUID PROPERTIES AND MIXES
#=========================================================
"""
Fluid properties.

These functions implement equations from Batzle and Wang (1992), seismic
properties of pore fluids. GEOPHYSICS, VOL. 57, NO. 11; P. 1396-1408,

:copyright: 2018 Agile Geoscience
:license: Apache 2.0
"""
def wood(Kf1, Kf2, Sf1):
    """
    Wood's equation, per equation 35b in Batzle and Wang (1992).
    """
    return 1 / ((Sf1 / Kf1) + ((1 - Sf1) / Kf2))


def rho_water(temperature, pressure):
    """
    The density of pure water, as a function of temperature and pressure.
    Implements equation 27a from Batzle and Wang (1992).

    Use scalars or arrays; if you use arrays, they must be the same size.

    Args:
        temperature (array): The temperature in degrees Celsius.
        pressure (array): The pressure in pascals.

    Returns:
        array: The density in kg/m3.
    """
    # Align with the symbols and units in Batzle & Wang.
    T, P = np.asanyarray(temperature), np.asanyarray(pressure)*1e-6

    x = -80*T - 3.3*T**2 + 0.00175*T**3 + 489*P - 2*T*P \
        + 0.016*P*T**2 - 1.3e-5*P*T**3 - 0.333*P**2 - 0.002*T*P**2

    return 1000 + 1e-3 * x


def rho_brine(temperature, pressure, salinity):
    """
    The density of NaCl brine, given temperature, pressure, and salinity.
    The density of pure water is computed from rho_water(). Implements
    equation 27b from Batzle and Wang (1992).

    Use scalars or arrays; if you use arrays, they must be the same size.

    Args:
        temperature (array): The temperature in degrees Celsius.
        pressure (array): The pressure in pascals.
        salinity (array): The weight fraction of NaCl, e.g. 35e-3
            for 35 parts per thousand, or 3.5% (the salinity of
            seawater).
    Returns:
        array: The density in kg/m3.
    """
    # Align with the symbols and units in Batzle & Wang.
    T, P = np.asanyarray(temperature), np.asanyarray(pressure)*1e-6
    S = np.asanyarray(salinity)

    rho_w = rho_water(temperature, pressure) / 1000
    x = 300*P - 2400*P*S + T*(80 + 3*T - 3300*S - 13*P + 47*P*S)

    return rho_w + S*(0.668 + 0.44*S + 1e-6 * x)


def rho_gas(temperature, pressure, molecular_weight):
    """
    Gas density, given temperature (in deg C), pressure (in Pa), and molecular
    weight.

    Args:
        temperature (array): The temperature in degrees Celsius.
        pressure (array): The pressure in pascals.
        molecular_weight (array): The molecular weight.
    Returns:
        array: The density in kg/m3.
    """
    # Align with the symbols and units in Batzle & Wang.
    T, P = np.asanyarray(temperature), np.asanyarray(pressure)*1e-6
    M = np.asanyarray(molecular_weight)
    R = 8.3144598

    return M * P / (R * (T + 273.15))


def v_water(temperature, pressure):
    """
    The acoustic velocity of pure water, as a function of temperature
    and pressure. Implements equation 28 from Batzle and Wang (1992), using
    the coefficients in Table 1.

    Note that this function does not work at pressures above about 100 MPa.

    Use scalars or arrays; if you use arrays, they must be the same size.

    Args:
        temperature (array): The temperature in degrees Celsius.
        pressure (array): The pressure in pascals.

    Returns:
        array: The velocity in m/s.
    """
    w = np.array([[ 1.40285e+03,  1.52400e+00,  3.43700e-03, -1.19700e-05],
                  [ 4.87100e+00, -1.11000e-02,  1.73900e-04, -1.62800e-06],
                  [-4.78300e-02,  2.74700e-04, -2.13500e-06,  1.23700e-08],
                  [ 1.48700e-04, -6.50300e-07, -1.45500e-08,  1.32700e-10],
                  [-2.19700e-07,  7.98700e-10,  5.23000e-11, -4.61400e-13]])

    T, P = np.asanyarray(temperature), np.asanyarray(pressure) * 1e-6
    return sum(w[i, j] * T**i * P**j for i in range(5) for j in range(4))


def v_brine(temperature, pressure, salinity):
    """
    The acoustic velocity of brine, as a function of temperature (deg C),
    pressure (Pa), and salinity (weight fraction). Implements equation 29
    from Batzle and Wang (1992).

    Note that this function does not work at pressures above about 100 MPa.

    Use scalars or arrays; if you use arrays, they must be the same size.

    Args:
        temperature (array): The temperature in degrees Celsius.
        pressure (array): The pressure in pascals.
        salinity (array): The weight fraction of NaCl, e.g. 35e-3
            for 35 parts per thousand, or 3.5% (the salinity of
            seawater).

    Returns:
        array: The velocity in m/s.
    """
    # Align with the symbols and units in Batzle & Wang.
    T, P = np.asanyarray(temperature), np.asanyarray(pressure)*1e-6
    S = np.asanyarray(salinity)

    v_w = v_water(temperature, pressure)
    s1 = 1170 - 9.6*T + 0.055*T**2 - 8.5e-5*T**3 + 2.6*P - 0.0029*T*P - 0.0476*P**2
    s15 = 780 - 10*P + 0.16*P**2
    s2 = -820

    return v_w + s1 * S + s15 * S**1.5 + s2 * S**2

#=========================================================
# FLUID SUBSTITUTIONS
#=========================================================

"""
:copyright: 2015 Agile Geoscience
:license: Apache 2.0

===================
fluidsub.py
===================

Calculates various parameters for fluid substitution
from Vp, Vs, and rho

Matt Hall, Evan Bianco, July 2014

Using http://www.subsurfwiki.org/wiki/Gassmann_equation

The algorithm is from Avseth et al (2006), per the wiki page.

Informed by Smith et al, Geophysics 68(2), 2003.

At some point we should do Biot too, per Russell...
http://cseg.ca/symposium/archives/2012/presentations/Biot_Gassmann_and_me.pdf
"""
def avseth_gassmann(ksat1, kf1, kf2, k0, phi):
    """
    Applies the inverse Gassmann's equation to calculate the rock bulk modulus
    saturated with fluid with bulk modulus. Requires as inputs the insitu
    fluid bulk modulus, the insitu saturated rock bulk modulus, the mineral
    matrix bulk modulus and the porosity.

    Args:
        ksat1 (float): initial saturated rock bulk modulus.
        kf1 (float): initial fluid bulk modulus.
        kf2 (float): final fluid bulk modulus.
        k0 (float): mineral bulk modulus.
        phi (float): porosity.

    Returns:
        ksat2 (float): final saturated rock bulk modulus.
    """

    s = ksat1 / (k0 - ksat1)
    f1 = kf1 / (phi * (k0 - kf1))
    f2 = kf2 / (phi * (k0 - kf2))
    ksat2 = k0 / ((1/(s - f1 + f2)) + 1)

    return ksat2


def smith_gassmann(kdry, k0, kf, phi):
    """
    Applies the direct Gassmann's equation to calculate the saturated rock
    bulk modulus from porosity and the dry-rock, mineral and fluid bulk moduli.

    Args:
        kdry (float): dry-rock bulk modulus.
        k0 (float): mineral bulk modulus.
        kf (float): fluid bulk modulus.
        phi (float): Porosity.

    Returns:
        ksat (float): saturated rock bulk modulus.
    """

    a = (1 - kdry/k0)**2.0
    b = phi/kf + (1-phi)/k0 - (kdry/k0**2.0)
    ksat = kdry + (a/b)

    return ksat


def vrh(kclay, kqtz, vclay):
    """
    Voigt-Reuss-Hill average to find Kmatrix from clay and qtz components.
    Works for any two components, they don't have to be clay and quartz.

    From Smith et al, Geophysics 68(2), 2003.

    Args:
        kclay (float): K_clay.
        kqtz (float): K_quartz.
        vclay (float): V_clay.

    Returns:
        Kvrh, also known as Kmatrix.
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
    Naive fluid substitution from Avseth et al, section 1.3.1. Bulk modulus of
    fluid 1 and fluid 2 are already known, and the bulk modulus of the dry
    frame, Kmin, is known. Use SI units.

    Args:
        vp (float): P-wave velocity
        vs (float): S-wave velocity
        rho (float): bulk density
        phi (float): porosity (volume fraction, e.g. 0.20)
        rhof1 (float): bulk density of original fluid (base case)
        rhof2 (float): bulk density of substitute fluid (subbed case)
        kmin (float): bulk modulus of solid mineral (s)
        kf1 (float): bulk modulus of original fluid
        kf2 (float): bulk modulus of substitute fluid

    Returns:
        Tuple: Vp, Vs, and rho for the substituted case
    """

    # Step 1: Extract the dynamic bulk and shear moduli.
    ksat1 = moduli.bulk(vp=vp, vs=vs, rho=rho)
    musat1 = moduli.mu(vp=vp, vs=vs, rho=rho)

    # Step 2: Apply Gassmann's relation.
    ksat2 = avseth_gassmann(ksat1=ksat1, kf1=kf1, kf2=kf2, k0=kmin, phi=phi)

    # Step 3: Leave the shear modulus unchanged.
    musat2 = musat1

    # Step 4: Correct the bulk density for the change in fluid.
    rho2 = rho + phi * (rhof2 - rhof1)

    # Step 5: recompute the fluid substituted velocities.
    vp2 = moduli.vp(bulk=ksat2, mu=musat2, rho=rho2)
    vs2 = moduli.vs(mu=musat2, rho=rho2)

    FluidSubResult = namedtuple('FluidSubResult', ['Vp', 'Vs', 'rho'])
    return FluidSubResult(vp2, vs2, rho2)


def smith_fluidsub(vp, vs, rho, phi, rhow, rhohc,
                   sw, swnew, kw, khc, kclay, kqtz,
                   vclay,
                   rhownew=None, rhohcnew=None,
                   kwnew=None, khcnew=None
                   ):
    """
    Naive fluid substitution from Smith et al. 2003. No pressure/temperature
    correction. Only works for SI units right now.

    Args:
        vp (float): P-wave velocity
        vs (float): S-wave velocity
        rho (float): bulk density
        phi (float): porosity (v/v, fraction)
        rhow (float): density of water
        rhohc (float): density of HC
        sw (float): water saturation in base case
        swnew (float): water saturation in subbed case
        kw (float):  bulk modulus of water
        khc (float): bulk modulus of hydrocarbon
        kclay (float): bulk modulus of clay
        kqtz (float):  bulk modulus of quartz
        vclay (float): Vclay, v/v
        rhownew (float): density of water in subbed case (optional)
        rhohcnew (float): density of HC in subbed case (optional)
        kwnew (float):  bulk modulus of water in subbed case (optional)
        khcnew (float): bulk modulus of hydrocarbon in subbed case (optional)

    Returns:
        Tuple: Vp, Vs, and rho for the substituted case.
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
    ksat = moduli.bulk(vp=vp, vs=vs, rho=rho)
    g = moduli.mu(vs=vs, rho=rho)

    # Step 4. Calculate K0 based on lithology estimates (VRH or HS mixing).
    k0 = vrh(kclay=kclay, kqtz=kqtz, vclay=vclay)

    # Step 5. Calculate fluid properties (K and ρ).
    # Step 6. Mix fluids for the in-situ case according to Sw.
    kfl = fluids.wood(kw, khc, sw)
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
    ksat2 = smith_gassmann(kdry=kstar, k0=k0, kf=kfl2, phi=phi)

    # Step 10: Calculate the new bulk density.
    # First we need the grain density...
    rhog = (rho - phi*rhofl) / (1-phi)
    # Now we can find the new bulk density
    rho2 = phi*rhofl2 + rhog*(1-phi)

    # Step 11: Calculate the new compressional velocity.
    # Remember, mu (G) is unchanged.
    vp2 = moduli.vp(bulk=ksat2, mu=g, rho=rho2)

    # Step 12: Calculate the new shear velocity.
    vs2 = moduli.vs(mu=g, rho=rho2)

    # Finish.
    FluidSubResult = namedtuple('FluidSubResult', ['Vp', 'Vs', 'rho'])
    return FluidSubResult(vp2, vs2, rho2)


def vels(Kdry, Gdry, K0, D0, Kf, Df, phi):
    '''
    Calculate velocities and densities of saturated rock via Gassmann equation.
    Provide all quantities in SI units.

    Parameters
    ----------   
    Kdry, Gdry : float or array_like
        Dry rock bulk & shear modulus in Pa.
    K0, G0 : float or array_like
        Mineral bulk & shear modulus in Pa.
    Kf, Df : float or array_like
        Fluid bulk modulus in Pa and density in kg/m^3.
    phi : float or array_like
        Porosity, v/v.

    Returns
    -------
    vp, vs : float or array_like
        Saturated rock P- and S-wave velocities in m/s.
    rho: float or array_like
        Saturated rock density in kg/m^3.
    K : float or array_like
        Saturated rock bulk modulus in Pa.
    '''
    rho = D0 * (1 - phi) + Df * phi
    with np.errstate(divide='ignore', invalid='ignore'):
        K = Kdry + (1 - Kdry / K0)**2 / ( (phi / Kf)
            + ((1 - phi) / K0) - (Kdry / K0**2) )
        vp = np.sqrt((K + 4/3 * Gdry) / rho)
        vs = np.sqrt(Gdry / rho)
    return vp, vs, rho, K

#=========================================================
# MODULI CALCULATION
#=========================================================

"""
===================
moduli.py
===================

Converts between various acoustic/eslatic parameters, and provides a way to
calculate all the elastic moduli from Vp, Vs, and rho.

Created June 2014, Matt Hall

Using equations http://www.subsurfwiki.org/wiki/Elastic_modulus
from Mavko, G, T Mukerji and J Dvorkin (2003), The Rock Physics Handbook,
Cambridge University Press.
"""
def youngs(vp=None, vs=None, rho=None, mu=None, lam=None, bulk=None, pr=None,
           pmod=None):
    """
    Computes Young's modulus given either Vp, Vs, and rho, or
    any two elastic moduli (e.g. lambda and mu, or bulk and P
    moduli). SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, mu, bulk, pr, and pmod

    Returns:
        Young's modulus in pascals, Pa
    """
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
    """
    Computes bulk modulus given either Vp, Vs, and rho, or
    any two elastic moduli (e.g. lambda and mu, or Young's
    and P moduli). SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, mu, youngs, pr, and pmod

    Returns:
        Bulk modulus in pascals, Pa
    """
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
    """
    Computes Poisson ratio given either Vp, Vs, and rho, or
    any two elastic moduli (e.g. lambda and mu, or Young's
    and P moduli). SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, mu, youngs, bulk, and pmod

    Returns:
        Poisson's ratio, dimensionless
    """
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


def mu(vp=None, vs=None, rho=None, pr=None, lam=None, youngs=None, bulk=None,
       pmod=None):
    """
    Computes shear modulus given either Vp, Vs, and rho, or
    any two elastic moduli (e.g. lambda and bulk, or Young's
    and P moduli). SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, bulk, youngs, pr, and pmod

    Returns:
        Shear modulus in pascals, Pa
    """
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


def lam(vp=None, vs=None, rho=None, pr=None,  mu=None, youngs=None, bulk=None,
        pmod=None):
    """
    Computes lambda given either Vp, Vs, and rho, or
    any two elastic moduli (e.g. bulk and mu, or Young's
    and P moduli). SI units only.

    Args:
        vp, vs, and rho
        or any 2 from bulk, mu, youngs, pr, and pmod

    Returns:
        Lambda in pascals, Pa
    """
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


def pmod(vp=None, vs=None, rho=None, pr=None, mu=None, lam=None, youngs=None,
         bulk=None):
    """
    Computes P-wave modulus given either Vp, Vs, and rho, or
    any two elastic moduli (e.g. lambda and mu, or Young's
    and bulk moduli). SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, mu, youngs, pr, and bulk

    Returns:
        P-wave modulus in pascals, Pa
    """
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
    """
    Computes Vp given bulk density and any two elastic moduli
    (e.g. lambda and mu, or Young's and P moduli). SI units only.

    Args:
        Any 2 from lam, mu, youngs, pr, pmod, bulk
        Rho

    Returns:
        Vp in m/s
    """
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

    elif (bulk is not None) and (vs and rho is not None):
        return np.sqrt((bulk / rho) + (4/3)*(vs**2))

    else:
        return None


def vs(youngs=None, vp=None, rho=None, mu=None, lam=None, bulk=None, pr=None,
       pmod=None):
    """
    Computes Vs given bulk density and shear modulus. SI units only.

    Args:
        Mu
        Rho

    Returns:
        Vs in m/s
    """
    if (mu is not None) and (rho is not None):
        return np.sqrt(mu / rho)

    else:
        return None


def moduli_dict(vp, vs, rho):
    """
    Computes elastic moduli given Vp, Vs, and rho. SI units only.

    Args:
        Vp, Vs, and rho

    Returns:
        A dict of elastic moduli, plus P-wave impedance.
    """
    mod = {}

    mod['imp'] = vp * rho

    mod['mu'] = mu(vs=vs, rho=rho)
    mod['pr'] = pr(vp=vp, vs=vs, rho=rho)
    mod['lam'] = lam(vp=vp, vs=vs, rho=rho)
    mod['bulk'] = bulk(vp=vp, vs=vs, rho=rho)
    mod['pmod'] = pmod(vp=vp, rho=rho)
    mod['youngs'] = youngs(vp=vp, vs=vs, rho=rho)

    return mod

#=========================================================
# ROCK PHYSICS MODELS with VERNIK's SAND SHALE MODELS
#=========================================================
"""
====================
rockphysicsmodels.py
====================

A bunch of rock physics models.
References are mentioned in docstrings of individual functions.
Docstrings follow numpy/scipy convention.

Alessandro Amato del Monte, March 2019
"""
def critical_porosity(K0, G0, phi, phi_c=0.4):
    '''
    Critical porosity model.
    This model describes the elastic behaviour (K=bulk and G=shear moduli)
    of dry sand for porosities below the critical porosity (phi_c).
    Above phi_c, the fluid phase is load-bearing, below phi_c the
    solid phase (mineral grains) is load-bearing.
    The equations here describe the variation of K and G
    for porosities below phi_c as straight lines.
    Critical porosity is usually 0.4 for sandstone,
    0.7 for chalk, 0.02-0.03 for granites.

    Parameters
    ----------   
    K0, G0 : float or array_like
        Mineral bulk & shear modulus in GPa.
    phi : float or array_like
        Porosity.
    phi_c : float, optional
        Critical porosity. Default: 0.4

    Returns
    -------
    Kdry, Gdry : float or array_like
        Dry rock bulk & shear modulus in GPa.

    References
    ----------
    Mavko et al. (2009), The Rock Physics Handbook, Cambridge University Press (p.370)
    '''
    Kdry = K0 * (1 - phi / phi_c)
    Gdry = G0 * (1 - phi / phi_c)
    return Kdry, Gdry


def hertz_mindlin(K0, G0, sigma, phi_c=0.4, Cn=8.6, f=1):
    '''
    Hertz-Mindlin model.
    This model describes the elastic behaviour (K=bulk and G=shear moduli)
    of a dry pack of spheres subject to a hydrostatic confining pressure. 

    Parameters
    ----------   
    K0, G0 : float or array_like
        Mineral bulk & shear modulus in GPa.
    phi : float or array_like
        Porosity.
    sigma : float
        Effective stress in MPa.
    phi_c : float, optional
        Critical porosity. Default: 0.4
    Cn : float, optional
        Coordination number Default: 8.6.
    f : float, optional
        Shear modulus correction factor,
        f=1 for dry pack with perfect adhesion
        between particles and f=0 for dry frictionless pack.

    Returns
    -------
    Kdry, Gdry : float or array_like
        Dry rock bulk & shear modulus in GPa.

    References
    ----------
    Mavko et al. (2009), The Rock Physics Handbook, Cambridge University Press (p.246)
    '''
    sigma0 = sigma / 1e3  # converts pressure in same units as solid moduli (GPa)
    pr0 =(3*K0-2*G0) / (6*K0+2*G0)  # poisson's ratio of mineral mixture
    Khm = (sigma0*(Cn**2*(1 - phi_c)**2*G0**2) / (18
        * np.pi**2 * (1 - pr0)**2))**(1/3)
    Ghm = ((2+3*f-pr0*(1+3*f)) / (5*(2-pr0))) * ((
        sigma0 * (3 * Cn**2 * (1 - phi_c)**2 * G0**2) / (
        2 * np.pi**2 * (1 - pr0)**2)))**(1/3)
    return Khm, Ghm


def soft_sand(K0, G0, phi, sigma, phi_c=0.4, Cn=8.6, f=1):
    '''
    Soft sand, or friable sand or uncemented sand model.
    This model describes the elastic behaviour (K=bulk and G=shear moduli)
    of poorly sorted dry sand by interpolating with the lower Hashin-Shtrikman bound
    the two end members at zero porosity and critical porosity.
    The zero porosity end member has K and G equal to mineral.
    The end member at critical porosity has K and G given by Hertz-Mindlin model.

    Parameters
    ----------   
    K0, G0 : float or array_like
        Mineral bulk & shear modulus in GPa.
    phi : float or array_like
        Porosity.
    sigma : float
        Effective stress in MPa.
    phi_c : float, optional
        Critical porosity. Default: 0.4
    Cn : float, optional
        Coordination number Default: 8.6.
    f : float, optional
        Shear modulus correction factor,
        f=1 for dry pack with perfect adhesion
        between particles and f=0 for dry frictionless pack.

    Returns
    -------
    Kdry, Gdry : float or array_like
        Dry rock bulk & shear modulus in GPa.

    References
    ----------
    Mavko et al. (2009), The Rock Physics Handbook, Cambridge University Press (p.258)
    '''
    Khm, Ghm = hertz_mindlin(K0, G0, sigma, phi_c, Cn, f)
    Kdry = -4/3 * Ghm + (((phi / phi_c) / (Khm + 4/3 * Ghm))
        + ((1 - phi / phi_c) / (K0 + 4/3 * Ghm)))**-1
    gxx = Ghm / 6 * ((9 * Khm + 8 * Ghm)  /  (Khm + 2 * Ghm))
    Gdry = -gxx + ((phi / phi_c) / (Ghm + gxx)
        + ((1 - phi / phi_c) / (G0 + gxx)))**-1
    return Kdry, Gdry


def stiff_sand(K0, G0, phi, sigma, phi_c=0.4, Cn=8.6, f=1):
    '''
    Stiff sand model.
    This model describes the elastic behaviour (K=bulk and G=shear moduli)
    of stiff dry sands by interpolating with the upper Hashin-Shtrikman bound
    the two end members at zero porosity and critical porosity.
    The zero porosity end member has K and G equal to mineral.
    The end member at critical porosity has K and G given by Hertz-Mindlin model.

    Parameters
    ----------   
    K0, G0 : float or array_like
        Mineral bulk & shear modulus in GPa.
    phi : float or array_like
        Porosity.
    sigma : float
        Effective stress in MPa.
    phi_c : float, optional
        Critical porosity. Default: 0.4
    Cn : float, optional
        Coordination number Default: 8.6.
    f : float, optional
        Shear modulus correction factor,
        f=1 for dry pack with perfect adhesion
        between particles and f=0 for dry frictionless pack.  
    
    Returns
    -------
    Kdry, Gdry : float or array_like
        Dry rock bulk & shear modulus in GPa

    References
    ----------
    Mavko et al. (2009), The Rock Physics Handbook, Cambridge University Press (p.260)
    '''
    Khm, Ghm = hertz_mindlin(K0, G0, sigma, phi_c, Cn, f)
    Kdry = -4/3 * G0 + (((phi / phi_c) / (Khm + 4/3 * G0))
        + ((1 - phi / phi_c) / (K0 + 4/3 * G0)))**-1
    tmp = G0 / 6*((9 * K0 + 8 * G0)  /  (K0 + 2 * G0))
    Gdry = -tmp + ((phi / phi_c) / (Ghm + tmp)
        + ((1 - phi / phi_c) / (G0 + tmp)))**-1
    return Kdry, Gdry


def contact_cement(K0, G0, phi, phi_c=0.4, Cn=8.6, Kc=37, Gc=45, scheme=2):
    '''
    Contact cement or cemented sand model,.
    This model describes the elastic behaviour (K=bulk and G=shear moduli)
    of dry sand where cement is deposited at grain contacts.
    The cement properties can be modified as well as the type of 
    cementation scheme.

    Parameters
    ----------
    K0, G0 : float or array_like
        Mineral bulk & shear modulus in GPa.
    phi : float or array_like
        Porosity.
    phi_c : float, optional
        Critical porosity. Default: 0.4
    Cn : float, optional
        Coordination number Default: 8.6.
    Kc, Gc : float, optional
        Cement bulk & shear modulus in GPa. Default: 37, 45.
    scheme : int, optional
        Cementation scheme, can be either 1 or 2:
        1: cement deposited at grain contacts
        2: cement as uniform layer around grains.
        Default: 2.

    Returns
    -------
    Kdry, Gdry : float or array_like
        Dry rock bulk & shear modulus in GPa.

    References
    ----------
    Dvorkin-Nur (1996), Elasticity of High-Porosity Sandstones: Theory for Two North Sea Data Sets.
    Geophysics 61, no. 5 (1996).
    Mavko et al. (2009), The Rock Physics Handbook, Cambridge University Press (p.255)
    '''
    pr0 = (3 * K0 - 2 * G0) / (6 * K0 + 2 * G0)
    PRc = (3 * Kc - 2 * Gc) / (6 * Kc + 2 * Gc)
    if scheme == 1:  # scheme 1: cement deposited at grain contacts
        alpha = ((phi_c - phi) / (3 * Cn * (1 - phi_c)))**(1/4)
    else:  # scheme 2: cement evenly deposited on grain surface
        alpha = ((2 * (phi_c - phi)) / (3 * (1 - phi_c)))**(1/2)
    LambdaN = (2 * Gc * (1 - pr0) * (1 - PRc)) / (
        np.pi * G0 * (1 - 2 * PRc))
    N1 = -0.024153 * LambdaN**-1.3646
    N2 = 0.20405 * LambdaN**-0.89008
    N3 = 0.00024649 * LambdaN**-1.9864
    Sn = N1 * alpha**2 + N2 * alpha + N3
    LambdaT = Gc / (np.pi * G0)
    T1 = -10**-2 * (2.26 * pr0**2 + 2.07 * pr0
        + 2.3) * LambdaT**(0.079 * pr0**2 + 0.1754 * pr0 - 1.342)
    T2 = (0.0573 * pr0**2 + 0.0937 * pr0
        + 0.202) * LambdaT**(0.0274 * pr0**2 + 0.0529 * pr0 - 0.8765)
    T3 = 10**-4 * (9.654 * pr0**2 + 4.945 * pr0
        + 3.1) * LambdaT**(0.01867 * pr0**2 + 0.4011 * pr0 - 1.8186)
    St = T1 * alpha**2 + T2 * alpha + T3
    Kdry = 1 / 6 * Cn *(1 - phi_c ) * (Kc + 4/3 * Gc) * Sn
    Gdry = 3 / 5 * Kdry + 3 / 20 * Cn * (1 - phi_c) * Gc * St
    return Kdry, Gdry


def constant_cement(K0, G0, phi, phi_cem=0.38, phi_c=0.4, Cn=8.6, Kc=37, Gc=45, scheme=2):
    '''
    Constant cement model, Avseth et al. (2000).
    This model describes the elastic behaviour (K=bulk and G=shear moduli)
    of high porosity dry sand with a certain initial cementation
    by interpolating with the lower Hashin-Shtrikman bound
    the two end members at zero porosity and critical porosity.
    The zero porosity end member has K and G equal to mineral.
    The high porosity end member has K and G given by the Contact Cement model.
    It is assumed that the porosity reduction is due to
    non-cementing material filling in the available pore space.

    Parameters
    ----------
    K0, G0 : float or array_like
        Mineral bulk & shear modulus in GPa.
    phi : float or array_like
        Porosity.
    phi_cem : float, optional
        Porosity at initial cementation. Default: 0.38.
    phi_c : float, optional
        Critical porosity. Default: 0.4.
    Cn : float, optional
        Coordination number. Default: 8.6.
    Kc, Gc : float, optional
        Cement bulk & shear modulus in GPa. Default: 37, 45.
    scheme : int, optional
        Cementation scheme, can be either 1 or 2:
        1: cement deposited at grain contacts
        2: cement as uniform layer around grains.
        Default: 2.

    Returns
    -------
    Kdry, Gdry : float or array_like
        Dry rock bulk & shear modulus in GPa.
    
    References
    ----------
    Dvorkin et al. (2014), Seismic Reflections of Rock Properties, Cambridge University Press (p.30-31)
    '''
    # contact cement model
    Khi, Ghi = contact_cement(K0, G0, phi, phi_c=phi_c, Cn=Cn, Kc=Kc, Gc=Gc, scheme=scheme)
    # lower bound Hashin-Shtrikman starting from phi_cem
    Kcc, Gcc = contact_cement(K0, G0, phi_cem, phi_c=phi_c, Cn=Cn, Kc=Kc, Gc=Gc, scheme=scheme)
    Klo = -4/3 * Gcc + (((phi / phi_cem) / (Kcc + 4/3 * Gcc)) + (
        (1 - phi / phi_cem) / (K0 + 4/3 * Gcc)))**-1
    tmp = Gcc / 6* ((9 * Kcc + 8 * Gcc)  /  (Kcc + 2 * Gcc))
    Glo = -tmp + ((phi / phi_cem) / (Gcc + tmp) + (
        (1 - phi / phi_cem) / (G0 + tmp)))**-1
   
    # initialize empty vectors for K and G dry
    Kdry, Gdry = (np.full(phi.size, np.nan) for _ in range(2))
    # for porosities>phi_cem use [K,G]_HI = contact cement model
    # for porosities<=phi_cem use [K,G]_LO = constant cement model
    Kdry[phi > phi_cem] = Khi[phi > phi_cem]
    Kdry[phi <= phi_cem] = Klo[phi <= phi_cem]
    Gdry[phi > phi_cem] = Ghi[phi > phi_cem]
    Gdry[phi <= phi_cem] = Glo[phi <= phi_cem]
    return Kdry, Gdry


def increasing_cement(K0, G0, phi, phi_cem=0.38, phi_c=0.4, Cn=8.6, Kc=37, Gc=45, scheme=2):
    '''
    Increasing cement model (Modified Hashin-Shtrikman upper bound).
    This model describes the elastic behaviour (K=bulk and G=shear moduli)
    of a dry sand with a certain initial cementation
    by interpolating with the upper Hashin-Shtrikman bound
    the two end members at zero porosity and critical porosity.
    The zero porosity end member has K and G equal to mineral.
    The high porosity end member has K and G given by the Contact Cement model.
    Probably best to avoid using if for porosities>phi_cem.
    Need to check references.

    Parameters
    ----------
    K0, G0 : float or array_like
        Mineral bulk & shear modulus in GPa.
    phi : float or array_like
        Porosity.
    phi_cem : float, optional
        Porosity at initial cementation. Default: 0.38.
    phi_c : float, optional
        Critical porosity. Default: 0.4.
    Cn : float, optional
        Coordination number. Default: 8.6.
    Kc, Gc : float, optional
        Cement bulk & shear modulus in GPa. Default: 37, 45.
    scheme : int, optional
        Cementation scheme, can be either 1 or 2:
        1: cement deposited at grain contacts
        2: cement as uniform layer around grains.
        Default: 2.

    Returns
    -------
    Kdry, Gdry : float or array_like
        dry rock bulk & shear modulus in GPa.
    '''
    Kcc, Gcc = contact_cement(K0, G0, phi_cem, phi_c=phi_c, Cn=Cn, Kc=Kc, Gc=Gc, scheme=scheme)
    Kdry = -4/3 * G0 + (((phi / phi_cem) / (Kcc + 4/3 * G0)) + (
        (1 - phi / phi_cem) / (K0 + 4/3 * G0)))**-1
    tmp = G0 / 6 * ((9 * K0 + 8 * G0)  /  (K0 + 2 * G0))
    Gdry = -tmp + ((phi / phi_cem) / (Gcc + tmp) + (
        (1 - phi / phi_cem) / (G0 + tmp)))**-1
    return Kdry, Gdry


def vernik_consol_sand(K0, G0, phi, sigma, b=10):
    '''
    Vernik & Kachanov Consolidated Sand Model.
    This model describes the elastic behaviour (K=bulk and G=shear moduli)
    of consolidated dry sand subject to a hydrostatic confining pressure
    as a continuous solid containing pores and cracks.

    Parameters
    ----------
    K0, G0 : float or array_like
        Mineral bulk & shear modulus in GPa.
    phi : float or array_like
        Porosity.
    sigma : float
        Effective stress in MPa.
    b : float, optional
        Slope parameter in pore shape empirical equation, range: 8-12.
        Default: 10.

    Returns
    -------
    Kdry, Gdry : float or array_like
        Dry rock bulk & shear modulus in GPa.
    
    References
    ----------
    Vernik & Kachanov (2010), Modeling elastic properties of siliciclastic rocks, Geophysics v.75 n.6
    '''
    # empirical pore shape factor:
    p = 3.6 + b * phi
    q = p # true if phi>0.03
    psf = phi / (1 - phi)  # psf = pore shape factor multiplier

    # matrix properties: assuming arenites with K=35.6 GPa, G=33 GPa, Poisson's ratio nu_m = 0.146
    nu_m = 0.146
    Avm = (16 * (1 - nu_m**2) ) / ( 9 * (1 - 2 * nu_m))       # nu_m=0.146 --> Avm=2.46
    Bvm = (32 * (1 - nu_m) * (5 - nu_m)) / (45 * (2 - nu_m))  # nu_m=0.146 --> Bvm=1.59

    # crack density: inversely correlated to effective stress
    eta0 = 0.3 + 1.6 * phi  # crack density at zero stress
    d = 0.07  # compaction coefficient
    d = 0.02 + 0.003 * sigma
    cd = (eta0 * np.exp(-d * sigma)) / (1 - phi)

    # note: the presence at denominator of the factor (1 - phi) in psf and cd is needed
    # to account for the interaction effects, i.e. the presence of pores raises the average stress
    # in the matrix increasing compliance contributions of pores and cracks
    # this correction is referred to as Mori-Tanaka's scheme.
    # in this way, the original model which is a NIA (non-interaction model)
    # is extended and becomes effectively a model which does take into account interactions.
    Kdry = K0 * (1 + p * psf + Avm * cd)**-1
    Gdry = G0 * (1 + q * psf + Bvm * cd)**-1
    return Kdry, Gdry


def vernik_soft_sand_1(K0, G0, phi, sigma, phi_c=0.36, phi_con=0.26, b=10, n=2.00, m=2.05):
    '''
    Vernik & Kachanov Soft Sand Model 1.
    This model describes the elastic behaviour (K=bulk and G=shear moduli)
    of dry sand modeled as a granular material.
    Only applicable for porosities between the low-porosity end-member
    (at the consolidation porosity phi_con) and the high-porosity
    end-member (at the critical porosity phi_c).
    The low-porosity end member is calculated with Vernik's
    Consolidated Sand Model.

    Parameters
    ----------
    K0, G0 : float or array_like
        Mineral bulk & shear modulus in GPa.
    phi : float or array_like
        Porosity.
    sigma : float
        Effective stress in MPa.
    phi_c : float, optional
        Critical porosity, range 0.30-0.42. Default: 0.36.
    phi_con : float, optional
        Consolidation porosity, range 0.22-0.30. Default: 0.26.
    b : float, optional
        Slope parameter in pore shape empirical equation, range: 8-12.
        Default: 10.
    n, m : float
        Empirical factors. Default: 2.00, 2.05.

    Returns
    -------
    Kdry, Gdry : float or array_like
        Dry rock bulk & shear modulus in GPa.
    
    References
    ----------
    Vernik & Kachanov (2010), Modeling elastic properties of siliciclastic rocks, Geophysics v.75 n.6
    '''
    if isinstance(phi, np.ndarray):
        phi_edit = phi.copy()
        phi_edit[(phi_edit < phi_con) | (phi_edit > phi_c)]=np.nan
    else:
        phi_edit = np.array(phi)
        if (phi_edit < phi_con) | (phi_edit > phi_c):
            return np.nan, np.nan
    M0 = K0 + 4/3 * G0
    K_con, G_con = vernik_consol_sand(K0, G0, phi_con, sigma, b)
    M_con = K_con + 4/3 * G_con
    T = (1 - (phi_edit - phi_con) / (phi_c - phi_con))
    Mdry = M_con * T**n
    Gdry = G_con * T**m
    Kdry = Mdry - 4/3 * Gdry
    return Kdry, Gdry


def vernik_soft_sand_2(K0, G0, phi, p=20, q=20):
    '''
    Vernik & Kachanov Soft Sand Model 2.
    This model describes the elastic behaviour (K=bulk and G=shear moduli)
    of dry sand modeled as a granular material.
    Applicable in the entire porosity range.

    Parameters
    ----------
    K0, G0 : float or array_like
        Mineral bulk & shear modulus in GPa.
    phi : float or array_like
        Porosity.
    p, q : float, optional
        Pore shape factor for K and G, range: 10-45.
        Default: 20.

    Returns
    -------
    Kdry, Gdry : float or array_like
        Dry rock bulk & shear modulus in GPa.
    
    References
    ----------
    Vernik & Kachanov (2010), Modeling elastic properties of siliciclastic rocks, Geophysics v.75 n.6
    '''
    M0 = K0 + 4/3 * G0
    Mdry = M0 * (1 + p * (phi / (1 - phi)))**-1
    Gdry = G0 * (1 + q * (phi / (1 - phi)))**-1
    Kdry = Mdry - 4/3 * Gdry
    return Kdry, Gdry


def vernik_sand_diagenesis(K0, G0, phi, sigma, phi_c=0.36, phi_con=0.26, b=10, n=2.00, m=2.05):
    '''
    Vernik & Kachanov Sandstone Diagenesis Model.
    This model describes the elastic behaviour (K=bulk and G=shear moduli)
    of dry sand modeled as a continuous solid containing pores and cracks
    for porosities below phi_con (consolidation porosity)
    using Vernik's Consolidated Sand Model,  and as a granular material
    for porosities above phi_con using Vernik's Soft Sand Model 1.

    Parameters
    ----------
    K0, G0 : float or array_like
        Mineral bulk & shear modulus in GPa.
    phi : float or array_like
        Porosity.
    sigma : float
        Effective stress in MPa.
    phi_c : float, optional
        Critical porosity, range 0.30-0.42. Default: 0.36.
    phi_con : float, optional
        Consolidation porosity, range 0.22-0.30. Default: 0.26.
    b : float, optional
        Slope parameter in pore shape empirical equation, range: 8-12.
        Default: 10.
    n, m : float
        Empirical factors. Default: 2.00, 2.05.

    Returns
    -------
    Kdry, Gdry : float or array_like
        Dry rock bulk & shear modulus in GPa.

    References
    ----------
    Vernik & Kachanov (2010), Modeling elastic properties of siliciclastic rocks, Geophysics v.75 n.6
    '''
    Kdry, Gdry = vernik_consol_sand(K0, G0, phi, sigma, b)
    Kdry_soft, Gdry_soft = vernik_soft_sand_1(K0, G0, phi, sigma, phi_c, phi_con, b, n, m)
    if isinstance(phi, np.ndarray):
        uu = phi>=phi_con
        Kdry[uu] = Kdry_soft[uu]
        Gdry[uu] = Gdry_soft[uu]
        return Kdry, Gdry
    else:
        if phi <= phi_con:
            return Kdry, Gdry
        else:
            return Kdry_soft, Gdry_soft


def vernik_shale(vclay, phi, rhom=2.73, rhob=1, Mqz=96, c33_clay=33.4, A=0.00284):
    '''
    Vernik & Kachanov Shale Model.
    This model describes the elastic behaviour in terms of velocities
    and density of inorganic shales.

    Parameters
    ----------
    vclay : float or array_like
        Dry clay content volume fraction.
    phi : float or array_like
        Porosity, maximum 0.40.
    rhom : float, optional
        Shale matrix density in g/cc. Default: 2.73.
    rhob : float, optional
        Brine density in g/cc. Default: 1.
    Mqz : float, optional
        P-wave elastic modulus of remaining minerals in GPa
        Default: 96.
    c33_clay : float, optional
        Anisotropic clay constant in GPa. Default: 33.4.
    A : float, optional
        Empirical coefficient for Vs. Default is good for illite/smectite/chlorite,
        can be raised up to .006 for kaolinite-rich clays.
        Default: 0.00284.
        
    Returns
    -------
    vp, vs, density : float or array_like
        P- and S-wave velocities in m/s, density in g/cc.

    Notes
    -----
    Shale matrix density (rhom) averages 2.73 +/- 0.03 g/cc at porosities below 0.25.
    It gradually varies with compaction and smectite-to-illite transition.
    A more accurate estimate can be calculated with this equation:
    rhom = 2.76+0.001*((rho-2)-230*np.exp(-4*(rho-2)))

    References
    ----------
    Vernik & Kachanov (2010), Modeling elastic properties of siliciclastic rocks, Geophysics v.75 n.6
    '''
    rho_matrix = 2.65 * (1 - vclay) + rhom * vclay
    k = 5.2 - 1.3 * vclay
    B, C = 0.287, 0.79
    c33_min = (vclay / c33_clay + (1 - vclay) / Mqz)**-1
    c33 = c33_min * (1 - phi)**k
    vp = np.sqrt(c33 / (rhom * (1 - phi) + rhob * phi))
    vs = np.sqrt(A * vp**4 + B * vp**2 - C)
    rho = rho_matrix * (1 - phi) + rhob * phi
    return vp * 1e3,vs * 1e3, rho

#=========================================================
# RP TEMPLATE(EXAMPLE by AADM)
#=========================================================

def rpt(model='soft',vsh=0.0,fluid='gas',phic=0.4,Cn=8,P=10,f=1,cement='quartz'):
    if cement=='quartz':
        Kc, Gc = 37, 45
    elif cement=='calcite':
        Kc, Gc = 76.8, 32
    elif cement=='clay':
        Kc, Gc = 21, 7
    phi=np.linspace(0.1,phic-.1,6)
    sw=np.linspace(0,1,5)
    (Khc, Dhc) = (Kg, Dg) if fluid == 'gas' else (Ko,Do)
    K0,G0 = vrh([vsh, 1-vsh],[Ksh,Kqz],[Gsh,Gqz])[4:]
    D0 = vsh*Dsh+(1-vsh)*Dqz
    if model=='soft':
        Kdry, Gdry = rpm.softsand(K0,G0,phi,phic,Cn,P,f)
    elif model=='stiff':
         Kdry, Gdry = rpm.stiffsand(K0,G0,phi,phic,Cn,P,f)
    elif model=='cem':
         Kdry, Gdry = rpm.contactcement(K0,G0,phi,phic,Cn,Kc,Gc,scheme=2)
    elif model=='crit':
         Kdry, Gdry = rpm.critpor(K0,G0,phi,phic)

    xx=np.empty((phi.size,sw.size))
    yy=np.empty((phi.size,sw.size))

    for i,val in enumerate(sw):
        Kf = vrh([val,1-val],[Kb,Khc],[999,999])[1]
        Df = val*Db+(1-val)*Dhc
        vp,vs,rho,_= fluidsub.vels(Kdry,Gdry,K0,D0,Kf,Df,phi)
        xx[:,i]=vp*rho
        yy[:,i]=vp/vs

    plt.figure(figsize=(10,6))
    plt.plot(xx, yy, '-ok', alpha=0.3)
    plt.plot(xx.T, yy.T, '-ok', alpha=0.3)
    for i,val in enumerate(phi):
        plt.text(xx[i,-1],yy[i,-1]+.01,'$\phi={:.02f}$'.format(val), backgroundcolor='0.9')
    plt.text(xx[-1,0]-100,yy[-1,0],'$S_w={:.02f}$'.format(sw[0]),ha='right', backgroundcolor='0.9')
    plt.text(xx[-1,-1]-100,yy[-1,-1],'$S_w={:.02f}$'.format(sw[-1]),ha='right', backgroundcolor='0.9')
    plt.xlabel('Ip'), plt.ylabel('Vp/Vs')
    plt.xlim(xx.min()-xx.min()*.1,xx.max()+xx.max()*.1)
    plt.ylim(yy.min()-yy.min()*.1,yy.max()+yy.max()*.1)
    plt.title('RPT (N:G={0}, fluid={1})'.format(1-vsh, fluid))


#================================================================================================
# REFLECTION
#================================================================================================

# -*- coding: utf-8 -*-
"""
Various reflectivity algorithms.

:copyright: 2018 Agile Geoscience
:license: Apache 2.0
"""
def critical_angles(vp1, vp2, vs2=None):
    """Compute critical angle at an interface

    Given the upper (vp1) and lower (vp2) velocities at an interface. If you want the PS-wave critical angle as well,
    pass vs2 as well.

    Args:
        vp1 (ndarray): Upper layer P-wave velocity.
        vp2 (ndarray): Lower layer P-wave velocity.
        vs2 (ndarray): Lower layer S-wave velocity (optional).

    Returns:
        tuple: The first and second critical angles at the interface, in
            degrees. If there isn't a critical angle, it returns np.nan.
    """
    ca1 = ca2 = np.nan

    if vp1 < vp2:
        ca1 = np.degrees(np.arcsin(vp1/vp2))

    if (vs2 is not None) and (vp1 < vs2):
        ca2 = np.degrees(np.arcsin(vp1/vs2))

    return ca1, ca2


def reflection_phase(reflectivity):
    """
    Compute the phase of the reflectivity. Returns an array (or float) of
    the phase, in positive multiples of 180 deg or pi rad. So 1 is opposite
    phase. A value of 1.1 would be +1.1 \times \pi rad.

    Args:
        reflectivity (ndarray): The reflectivity, eg from `zoeppritz()`.

    Returns:
        ndarray: The phase, strictly positive
    """
    ph = np.arctan2(np.imag(reflectivity), np.real(reflectivity)) / np.pi
    ph[ph == 1] = 0
    ph[ph < 0] = 2 + ph[ph < 0]
    return ph


def acoustic_reflectivity(vp, rho):
    """
    The acoustic reflectivity, given Vp and RHOB logs.

    Args:
        vp (ndarray): The P-wave velocity.
        rho (ndarray): The bulk density.

    Returns:
        ndarray: The reflectivity coefficient series.
    """
    upper = vp[:-1] * rho[:-1]
    lower = vp[1:] * rho[1:]
    return (lower - upper) / (lower + upper)

def reflectivity(vp, vs, rho, theta=0, method='zoeppritz_rpp'):
    """
    Offset reflectivity, given Vp, Vs, rho, and offset.

    Computes 'upper' and 'lower' intervals from the three provided arrays,
    then passes the result to the specified method to compute reflection
    coefficients.

    For acoustic reflectivity, either use the `acoustic_reflectivity()`
    function, or call `reflectivity()` passing any log as Vs, e.g. just give
    the Vp log twice (it won't be used anyway):

        reflectivity(vp, vp, rho)

    For anisotropic reflectivity, use either `anisotropy.blangy()` or
    `anisotropy.ruger()`.

    Args:
        vp (ndarray): The P-wave velocity; float or 1D array length m.
        vs (ndarray): The S-wave velocity; float or 1D array length m.
        rho (ndarray): The density; float or 1D array length m.
        theta (ndarray): The incidence angle; float or 1D array length n.
        method (str): The reflectivity equation to use; one of:

                - 'scattering_matrix': scattering_matrix
                - 'zoeppritz_element': zoeppritz_element
                - 'zoeppritz': zoeppritz
                - 'zoeppritz_rpp': zoeppritz_rpp
                - 'akirichards': akirichards
                - 'akirichards_alt': akirichards_alt
                - 'fatti': fatti
                - 'shuey': shuey
                - 'bortfeld': bortfeld
                - 'hilterman': hilterman

        Notes:

                - scattering_matrix gives the full solution
                - zoeppritz_element gives a single element which you specify
                - zoeppritz returns RPP element only; use zoeppritz_rpp instead
                - zoeppritz_rpp is faster than zoeppritz or zoeppritz_element

    Returns:
        ndarray. The result of running the specified method on the inputs.
            Will be a float (for float inputs and one angle), a 1 x n array
            (for float inputs and an array of angles), a 1 x m-1 array (for
            float inputs and one angle), or an m-1 x n array (for array inputs
            and an array of angles).
    """
    methods = {
        'scattering_matrix': scattering_matrix,
        'zoeppritz_element': zoeppritz_element,
        'zoeppritz': zoeppritz,
        'zoeppritz_rpp': zoeppritz_rpp,
        'akirichards': akirichards,
        'akirichards_alt': akirichards_alt,
        'fatti': fatti,
        'shuey': shuey,
        'bortfeld': bortfeld,
        'hilterman': hilterman,
    }
    func = methods[method.lower()]
    vp = np.asanyarray(vp, dtype=float)
    vs = np.asanyarray(vs, dtype=float)
    rho = np.asanyarray(rho, dtype=float)

    vp1, vp2 = vp[:-1], vp[1:]
    vs1, vs2 = vs[:-1], vs[1:]
    rho1, rho2 = rho[:-1], rho[1:]

    return func(vp1, vs1, rho1, vp2, vs2, rho2, theta)


def vectorize(func):
    """
    Decorator to make sure the inputs are arrays. We also add a dimension
    to theta to make the functions work in an 'outer product' way.

    Takes a reflectivity function requiring Vp, Vs, and RHOB for 2 rocks
    (upper and lower), plus incidence angle theta, plus kwargs. Returns
    that function with the arguments transformed to ndarrays.
    """
    @wraps(func)
    def wrapper(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, **kwargs):
        vp1 = np.asanyarray(vp1, dtype=float)
        vs1 = np.asanyarray(vs1, dtype=float) + 1e-12  # Prevent singular matrix.
        rho1 = np.asanyarray(rho1, dtype=float)
        vp2 = np.asanyarray(vp2, dtype=float)
        vs2 = np.asanyarray(vs2, dtype=float) + 1e-12  # Prevent singular matrix.
        rho2 = np.asanyarray(rho2, dtype=float)
        theta1 = np.asanyarray(theta1).reshape((-1, 1))
        return func(vp1, vs1, rho1, vp2, vs2, rho2, theta1, **kwargs)
    return wrapper


def preprocess(func):
    """
    Decorator to preprocess arguments for the reflectivity equations.

    Takes a reflectivity function requiring Vp, Vs, and RHOB for 2 rocks
    (upper and lower), plus incidence angle theta, plus kwargs. Returns
    that function with some arguments transformed.
    """
    @wraps(func)
    def wrapper(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, **kwargs):

        # Interpret tuple for theta1 as a linspace.
        if isinstance(theta1, tuple):
            if len(theta1) == 2:
                start, stop = theta1
                theta1 = np.linspace(start, stop, num=stop+1)
            elif len(theta1) == 3:
                start, stop, step = theta1
                steps = (stop / step) + 1
                theta1 = np.linspace(start, stop, num=steps)
            else:
                raise TypeError("Expected 2 or 3 parameters for theta1 expressed as range.")

        # Convert theta1 to radians and complex numbers.
        theta1 = np.radians(theta1).astype(complex)

        return func(vp1, vs1, rho1, vp2, vs2, rho2, theta1, **kwargs)
    return wrapper


@preprocess
@vectorize
def scattering_matrix(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    """
    Full Zoeppritz solution, considered the definitive solution.
    Calculates the angle dependent p-wave reflectivity of an interface
    between two mediums.

    Originally written by: Wes Hamlyn, vectorized by Agile.

    Returns the complex reflectivity.

    Args:
        vp1 (float): The upper P-wave velocity.
        vs1 (float): The upper S-wave velocity.
        rho1 (float): The upper layer's density.
        vp2 (float): The lower P-wave velocity.
        vs2 (float): The lower S-wave velocity.
        rho2 (float): The lower layer's density.
        theta1 (ndarray): The incidence angle; float or 1D array length n.

    Returns:
        ndarray. The exact Zoeppritz solution for all modes at the interface.
            A 4x4 array representing the scattering matrix at the incident
            angle theta1.
    """
    theta1 *= np.ones_like(vp1)
    p = np.sin(theta1) / vp1  # Ray parameter.
    theta2 = np.arcsin(p * vp2)  # Trans. angle of P-wave.
    phi1 = np.arcsin(p * vs1)    # Refl. angle of converted S-wave.
    phi2 = np.arcsin(p * vs2)    # Trans. angle of converted S-wave.

    # Matrix form of Zoeppritz equations... M & N are matrices.
    M = np.array([[-np.sin(theta1), -np.cos(phi1), np.sin(theta2), np.cos(phi2)],
                  [np.cos(theta1), -np.sin(phi1), np.cos(theta2), -np.sin(phi2)],
                  [2 * rho1 * vs1 * np.sin(phi1) * np.cos(theta1),
                   rho1 * vs1 * (1 - 2 * np.sin(phi1) ** 2),
                   2 * rho2 * vs2 * np.sin(phi2) * np.cos(theta2),
                   rho2 * vs2 * (1 - 2 * np.sin(phi2) ** 2)],
                  [-rho1 * vp1 * (1 - 2 * np.sin(phi1) ** 2),
                   rho1 * vs1 * np.sin(2 * phi1),
                   rho2 * vp2 * (1 - 2 * np.sin(phi2) ** 2),
                   -rho2 * vs2 * np.sin(2 * phi2)]])

    N = np.array([[np.sin(theta1), np.cos(phi1), -np.sin(theta2), -np.cos(phi2)],
                  [np.cos(theta1), -np.sin(phi1), np.cos(theta2), -np.sin(phi2)],
                  [2 * rho1 * vs1 * np.sin(phi1) * np.cos(theta1),
                   rho1 * vs1 * (1 - 2 * np.sin(phi1) ** 2),
                   2 * rho2 * vs2 * np.sin(phi2) * np.cos(theta2),
                   rho2 * vs2 * (1 - 2 * np.sin(phi2) ** 2)],
                  [rho1 * vp1 * (1 - 2 * np.sin(phi1) ** 2),
                   -rho1 * vs1 * np.sin(2 * phi1),
                   - rho2 * vp2 * (1 - 2 * np.sin(phi2) ** 2),
                   rho2 * vs2 * np.sin(2 * phi2)]])

    M_ = np.moveaxis(np.squeeze(M), [0, 1], [-2, -1])
    A = np.linalg.inv(M_)
    N_ = np.moveaxis(np.squeeze(N), [0, 1], [-2, -1])
    Z_ = np.matmul(A, N_)

    return np.transpose(Z_, axes=list(range(Z_.ndim - 2)) + [-1, -2])


def zoeppritz_element(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, element='PdPu'):
    """
    Returns any mode reflection coefficients from the Zoeppritz
    scattering matrix. Pass in the mode as element, e.g. 'PdSu' for PS.

    Wraps scattering_matrix().

    Returns the complex reflectivity.

    Args:
        vp1 (float): The upper P-wave velocity.
        vs1 (float): The upper S-wave velocity.
        rho1 (float): The upper layer's density.
        vp2 (float): The lower P-wave velocity.
        vs2 (float): The lower S-wave velocity.
        rho2 (float): The lower layer's density.
        theta1 (ndarray): The incidence angle; float or 1D array length n.
        element (str): The name of the element to return, must be one of:
            'PdPu', 'SdPu', 'PuPu', 'SuPu', 'PdSu', 'SdSu', 'PuSu', 'SuSu',
            'PdPd', 'SdPd', 'PuPd', 'SuPd', 'PdSd', 'SdSd', 'PuSd', 'SuSd'.

    Returns:
        ndarray. Array length n of the exact Zoeppritz solution for the
            specified modes at the interface, at the incident angle theta1.
    """
    elements = np.array([['PdPu', 'SdPu', 'PuPu', 'SuPu'],
                         ['PdSu', 'SdSu', 'PuSu', 'SuSu'],
                         ['PdPd', 'SdPd', 'PuPd', 'SuPd'],
                         ['PdSd', 'SdSd', 'PuSd', 'SuSd']])

    Z = scattering_matrix(vp1, vs1, rho1, vp2, vs2, rho2, theta1).T

    return np.squeeze(Z[np.where(elements == element)].T)


def zoeppritz(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    """
    Returns the PP reflection coefficients from the Zoeppritz
    scattering matrix. Wraps zoeppritz_element().

    Returns the complex reflectivity.

    Args:
        vp1 (float): The upper P-wave velocity.
        vs1 (float): The upper S-wave velocity.
        rho1 (float): The upper layer's density.
        vp2 (float): The lower P-wave velocity.
        vs2 (float): The lower S-wave velocity.
        rho2 (float): The lower layer's density.
        theta1 (ndarray): The incidence angle; float or 1D array length n.

    Returns:
        ndarray. Array length n of the exact Zoeppritz solution for the
            specified modes at the interface, at the incident angle theta1.
    """
    return zoeppritz_element(vp1, vs1, rho1, vp2, vs2, rho2, theta1, 'PdPu')


@preprocess
@vectorize
def zoeppritz_rpp(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    """
    Exact Zoeppritz from expression.

    This is useful because we can pass arrays to it, which we can't do to
    scattering_matrix().

    Dvorkin et al. (2014). Seismic Reflections of Rock Properties. Cambridge.

    Returns the complex reflectivity.

    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        theta1 (ndarray): The incidence angle; float or 1D array length n.

    Returns:
        ndarray. The exact Zoeppritz solution for P-P reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """
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

    return np.squeeze(rpp)


@preprocess
@vectorize
def akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    The Aki-Richards approximation to the reflectivity.

    This is the formulation from Avseth et al., Quantitative seismic
    interpretation, Cambridge University Press, 2006. Adapted for a 4-term
    formula. See http://subsurfwiki.org/wiki/Aki-Richards_equation.

    Returns the complex reflectivity.

    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        theta1 (ndarray): The incidence angle; float or 1D array length n.
        terms (bool): Whether or not to return a tuple of the terms of the
            equation. The first term is the acoustic impedance.

    Returns:
        ndarray. The Aki-Richards approximation for P-P reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """
    theta2 = np.arcsin(vp2/vp1*np.sin(theta1))
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
    term2 = -1 * x * np.sin(theta1)**2
    term3 = y / np.cos(meantheta)**2
    term4 = -1 * z * np.sin(theta1)**2

    if terms:
        fields = ['term1', 'term2', 'term3', 'term4']
        AkiRichards = namedtuple('AkiRichards', fields)
        return AkiRichards(np.squeeze([term1 for _ in theta1]),
                           np.squeeze(term2),
                           np.squeeze(term3),
                           np.squeeze(term4)
                           )
    else:
        return np.squeeze(term1 + term2 + term3 + term4)


@preprocess
@vectorize
def akirichards_alt(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    This is another formulation of the Aki-Richards solution.
    See http://subsurfwiki.org/wiki/Aki-Richards_equation

    Returns the complex reflectivity.

    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        theta1 (ndarray): The incidence angle; float or 1D array length n.
        terms (bool): Whether or not to return a tuple of the terms of the
            equation. The first term is the acoustic impedance.

    Returns:
        ndarray. The Aki-Richards approximation for P-P reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """
    theta2 = np.arcsin(vp2/vp1*np.sin(theta1))
    drho = rho2-rho1
    dvp = vp2-vp1
    dvs = vs2-vs1
    theta = (theta1+theta2)/2.0
    rho = (rho1+rho2)/2.0
    vp = (vp1+vp2)/2.0
    vs = (vs1+vs2)/2.0

    # Compute the three terms
    term1 = 0.5 * (dvp/vp + drho/rho)
    term2 = (0.5*dvp/vp-2*(vs/vp)**2*(drho/rho+2*dvs/vs)) * np.sin(theta)**2
    term3 = 0.5 * dvp/vp * (np.tan(theta)**2 - np.sin(theta)**2)

    if terms:
        fields = ['term1', 'term2', 'term3']
        AkiRichards = namedtuple('AkiRichards', fields)
        return AkiRichards(np.squeeze([term1 for _ in theta1]),
                           np.squeeze(term2),
                           np.squeeze(term3)
                           )
    else:
        return np.squeeze(term1 + term2 + term3)


@preprocess
@vectorize
def fatti(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    Compute reflectivities with Fatti's formulation of the Aki-Richards
    equation, which does not account for the critical angle. See Fatti et al.
    (1994), Geophysics 59 (9). Real numbers only.

    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        theta1 (ndarray): The incidence angle; float or 1D array length n.
        terms (bool): Whether or not to return a tuple of the terms of the
            equation. The first term is the acoustic impedance.

    Returns:
        ndarray. The Fatti approximation for P-P reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """
    theta1 = np.real(theta1)

    drho = rho2-rho1
    rho = (rho1+rho2) / 2.0
    vp = (vp1+vp2) / 2.0
    vs = (vs1+vs2) / 2.0
    dip = (vp2*rho2 - vp1*rho1)/(vp2*rho2 + vp1*rho1)
    dis = (vs2*rho2 - vs1*rho1)/(vs2*rho2 + vs1*rho1)
    d = drho/rho

    # Compute the three terms
    term1 = (1 + np.tan(theta1)**2) * dip
    term2 = -8 * (vs/vp)**2 * dis * np.sin(theta1)**2
    term3 = -1 * (0.5 * np.tan(theta1)**2 - 2 * (vs/vp)**2 * np.sin(theta1)**2) * d

    if terms:
        fields = ['term1', 'term2', 'term3']
        Fatti = namedtuple('Fatti', fields)
        return Fatti(np.squeeze(term1),
                     np.squeeze(term2),
                     np.squeeze(term3)
                     )
    else:
        return np.squeeze(term1 + term2 + term3)


@preprocess
@vectorize
def shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0,
          terms=False,
          return_gradient=False):
    """
    Compute Shuey approximation with 3 terms.
    http://subsurfwiki.org/wiki/Shuey_equation

    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        theta1 (ndarray): The incidence angle; float or 1D array length n.
        terms (bool): Whether or not to return a tuple of the terms of the
            equation. The first term is the acoustic impedance.
        return_gradient (bool): Whether to return a tuple of the intercept
            and gradient (i.e. the second term divided by sin^2(theta)).

    Returns:
        ndarray. The Aki-Richards approximation for P-P reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """
    theta1 = np.real(theta1)

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
        return Shuey(np.squeeze(r0), np.squeeze(g))
    elif terms:
        fields = ['R0', 'Rg', 'Rf']
        Shuey = namedtuple('Shuey', fields)
        return Shuey(np.squeeze([term1 for _ in theta1]),
                     np.squeeze(term2),
                     np.squeeze(term3)
                     )
    else:
        return np.squeeze(term1 + term2 + term3)


@deprecated('Please use shuey() instead.')
def shuey2(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    """
    Compute Shuey approximation with 2 terms. Wraps `shuey()`. Deprecated,
    use `shuey()` instead.
    """
    r, g, _ = shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1=theta1, terms=True)
    return r + g


@deprecated('Please use shuey() instead.')
def shuey3(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    Compute Shuey approximation with 3 terms. Wraps `shuey()`. Deprecated,
    use `shuey()` instead.
    """
    return shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1=theta1)


@preprocess
@vectorize
def bortfeld(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    Compute Bortfeld approximation with three terms.
    http://sepwww.stanford.edu/public/docs/sep111/marie2/paper_html/node2.html
    Real numbers only.

    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        theta1 (ndarray): The incidence angle; float or 1D array length n.
        terms (bool): Whether or not to return a tuple of the terms of the
            equation. The first term is the acoustic impedance.

    Returns:
        ndarray. The 3-term Bortfeld approximation for P-P reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """
    theta1 = np.real(theta1)

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
        return Bortfeld(np.squeeze([term1 for _ in theta1]),
                        np.squeeze(term2),
                        np.squeeze(term3)
                        )
    else:
        return np.squeeze(term1 + term2 + term3)


@deprecated('Please use bortfeld() instead.')
def bortfeld2(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    The 2-term Bortfeld approximation for ava analysis. Wraps `shuey()`.
    Deprecated, use `bortfeld()` instead.

    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        theta1 (ndarray): The incidence angle; float or 1D array length n.
        terms (bool): Whether or not to return a tuple of the terms of the
            equation. The first term is the acoustic impedance.

    Returns:
        ndarray. The 2-term Bortfeld approximation for P-P reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """
    theta1 = np.radians(theta1)
    theta2 = np.arcsin(vp2/vp1*np.sin(theta1))
    term1 = 0.5 * np.log((vp2*rho2*np.cos(theta1)) / (vp1*rho1*np.cos(theta2)))
    svp2 = (np.sin(theta1)/vp1)**2
    dvs2 = (vs1**2-vs2**2)
    term2 = svp2 * dvs2 * (2+np.log(rho2/rho1)/np.log(vs2/vs1))

    if terms:
        return term1, term2
    else:
        return (term1 + term2)


@deprecated('Please use bortfeld() instead.')
def bortfeld3(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    return bortfeld(vp1, vs1, rho1, vp2, vs2, rho2, theta1=theta1)


@preprocess
@vectorize
def hilterman(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    Not recommended, only seems to match Zoeppritz to about 10 deg.

    Hilterman (1989) approximation from Mavko et al. Rock Physics Handbook.
    According to Dvorkin: "arguably the simplest and a very convenient
    [approximation]." At least for small angles and small contrasts. Real
    numbers only.

    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        theta1 (ndarray): The incidence angle; float or 1D array length n.
        terms (bool): Whether or not to return a tuple of the terms of the
            equation. The first term is the acoustic impedance.

    Returns:
        ndarray. The Hilterman approximation for P-P reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """
    theta1 = np.real(theta1)

    ip1 = vp1 * rho1
    ip2 = vp2 * rho2
    rp0 = (ip2 - ip1) / (ip2 + ip1)

    pr2, pr1 = moduli.pr(vp2, vs2), moduli.pr(vp1, vs1)
    pravg = (pr2 + pr1) / 2.
    pr = (pr2 - pr1) / (1 - pravg)**2.

    term1 = rp0 * np.cos(theta1)**2.
    term2 = pr * np.sin(theta1)**2.

    if terms:
        fields = ['term1', 'term2']
        Hilterman = namedtuple('Hilterman', fields)
        return Hilterman(np.squeeze(term1), np.squeeze(term2))
    else:
        return np.squeeze(term1 + term2)

#================================================================================================
# FILTERS, WAVELET, CONVOLUTION
#================================================================================================
# FILTERS
#=========================================================
"""
Smoothers.

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
class BrugesError(Exception):
    """
    Generic error class.
    """
    pass

def mean(arr, size=5):
    """
    A linear n-D smoothing filter. Can be used as a moving average on 1D data.

    Args:
        arr (ndarray): an n-dimensional array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5. Should be odd,
            rounded up if not.

    Returns:
        ndarray: the resulting smoothed array.
    """
    arr = np.array(arr, dtype=np.float)

    if not size // 2:
        size += 1

    return scipy.ndimage.generic_filter(arr, np.mean, size=size)


def rms(arr, size=5):
    """
    A linear n-D smoothing filter. Can be used as a moving average on 1D data.

    Args:
        arr (ndarray): an n-dimensional array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5. Should be odd,
            rounded up if not.

    Returns:
        ndarray: the resulting smoothed array.
    """
    arr = np.array(arr, dtype=np.float)

    if not size // 2:
        size += 1

    return scipy.ndimage.generic_filter(arr, rms_, size=size)


def median(arr, size=5):
    """
    A nonlinear n-D edge-preserving smoothing filter.

    Args:
        arr (ndarray): an n-dimensional array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5. Should be odd,
            rounded up if not.

    Returns:
        ndarray: the resulting smoothed array.
    """
    arr = np.array(arr, dtype=np.float)

    if not size // 2:
        size += 1

    return scipy.ndimage.generic_filter(arr, np.median, size=size)


def mode(arr, size=5, tie='smallest'):
    """
    A nonlinear n-D categorical smoothing filter. Use this to filter non-
    continuous variables, such as categorical integers, e.g. to label facies.

    Args:
        arr (ndarray): an n-dimensional array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5. Should be odd,
            rounded up if not.
        tie (str): `'smallest'` or `'largest`'. In the event of a tie (i.e. two
            or more values having the same count in the kernel), whether to
            give back the smallest of the tying values, or the largest.

    Returns:
        ndarray: the resulting smoothed array.
    """
    def func(this, tie):
        if tie == 'smallest':
            m, _ = scipy.stats.mode(this)
        else:
            m, _ = -scipy.stats.mode(-this)
        return np.squeeze(m)

    arr = np.array(arr, dtype=np.float)

    if not size // 2:
        size += 1

    return scipy.ndimage.generic_filter(arr, func, size=size,
                                        extra_keywords={'tie': tie}
                                       )


def snn(arr, size=5, include=True):
    """
    Symmetric nearest neighbour, a nonlinear 2D smoothing filter.
    http://subsurfwiki.org/wiki/Symmetric_nearest_neighbour_filter

    Args:
        arr (ndarray): a 2D array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5. Should be odd,
            rounded up if not.
        include (bool): whether to include the central pixel itself.

    Returns:
        ndarray: the resulting smoothed array.
    """
    def func(this, pairs, include):
        """
        Deal with this patch.
        """
        centre = this[this.size // 2]
        select = [nearest(this[p], centre) for p in pairs]
        if include:
            select += [centre]
        return np.mean(select)

    arr = np.array(arr, dtype=np.float)
    if arr.ndim != 2:
        raise BrugesError("arr must have 2-dimensions")

    if not size // 2:
        size += 1

    pairs = [[i, size**2-1 - i] for i in range(size**2 // 2)]
    return scipy.ndimage.generic_filter(arr,
                                        func,
                                        size=size,
                                        extra_keywords={'pairs': pairs,
                                                        'include': include}
                                       )


def kuwahara(arr, size=5):
    """
    Kuwahara, a nonlinear 2D smoothing filter.
    http://subsurfwiki.org/wiki/Kuwahara_filter

    Args:
        arr (ndarray): a 2D array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5. Should be odd,
            rounded up if not.

    Returns:
        ndarray: the resulting smoothed array.
    """
    def func(this, s, k):
        """
        Deal with this patch.
        """
        t = this.reshape((s, s))
        sub = np.array([t[:k, :k].flatten(),
                        t[:k, k-1:].flatten(),
                        t[k-1:, :k].flatten(),
                        t[k-1:, k-1:].flatten()]
                      )
        select = sub[np.argmin(np.var(sub, axis=1))]
        return np.mean(select)

    arr = np.array(arr, dtype=np.float)
    if arr.ndim != 2:
        raise BrugesError("arr must have 2-dimensions")

    if not size // 2:
        size += 1

    k = int(np.ceil(size / 2))

    return scipy.ndimage.generic_filter(arr,
                                        func,
                                        size=size,
                                        extra_keywords={'s': size,
                                                        'k': k,
                                                       }
                                       )


def conservative(arr, size=5, supercon=False):
    """
    Conservative, a nonlinear n-D despiking filter. Very conservative! Only
    changes centre value if it is outside the range of all the other values
    in the kernel. Read http://subsurfwiki.org/wiki/Conservative_filter

    Args:
        arr (ndarray): an n-dimensional array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5 (in a 2D arr). Should be
            odd, rounded up if not.
        supercon (bool): whether to be superconservative. If True, replaces
            pixel with min or max of kernel. If False (default), replaces pixel
            with mean of kernel.

    Returns:
        ndarray: the resulting smoothed array.
    """
    def func(this, k, supercon):
        this = this.flatten()
        centre = this[k]
        rest = [this[:k], this[-k:]]
        mi, ma = np.nanmin(rest), np.nanmax(rest)
        if centre < mi:
            return mi if supercon else np.mean(rest)
        elif centre > ma:
            return ma if supercon else np.mean(rest)
        else:
            return centre

    arr = np.array(arr, dtype=np.float)

    if not size // 2:
        size += 1

    k = int(np.floor(size**arr.ndim / 2))

    return scipy.ndimage.generic_filter(arr,
                                        func,
                                        size=size,
                                        extra_keywords={'k': k,
                                                        'supercon': supercon,
                                                       }
                                       )


def rotate_phase(s, phi, degrees=False):
    """
    Performs a phase rotation of wavelet or wavelet bank using:

    .. math::

        A = w(t)\cos(\phi) - h(t)\sin(\phi)

    where w(t) is the wavelet and h(t) is its Hilbert transform.

    The analytic signal can be written in the form S(t) = A(t)exp(j*theta(t))
    where A(t) = magnitude(hilbert(w(t))) and theta(t) = angle(hilbert(w(t))
    then a constant phase rotation phi would produce the analytic signal
    S(t) = A(t)exp(j*(theta(t) + phi)). To get the non analytic signal
    we take real(S(t)) == A(t)cos(theta(t) + phi)
    == A(t)(cos(theta(t))cos(phi) - sin(theta(t))sin(phi)) <= trig identity
    == w(t)cos(phi) - h(t)sin(phi)

    Args:
        w (ndarray): The wavelet vector, can be a 2D wavelet bank.
        phi (float): The phase rotation angle (in radians) to apply.
        degrees (bool): If phi is in degrees not radians.

    Returns:
        The phase rotated signal (or bank of signals).
    """
    # Make sure the data is at least 2D to apply_along
    data = np.atleast_2d(s)

    # Get Hilbert transform. This will be 2D.
    a = apply_along_axis(scipy.signal.hilbert, data, axis=0)

    # Transform angles into what we need.
    phi = np.asanyarray(phi).reshape(-1, 1, 1)
    if degrees:
        phi = np.radians(phi)
        
    rotated = np.real(a) * np.cos(phi)  -  np.imag(a) * np.sin(phi)
    return np.squeeze(rotated)

#=========================================================
# KERNELS (GAUSSIAN)
#=========================================================
"""
2D kernels for image processing.

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
def gaussian(size, size_y=None):
    """
    2D Gaussian Kernel

    Args:
        size (int): the kernel size, e.g. 5 for 5x5 (in a 2D arr). Should be
            odd, rounded up if not.
        size_y (int): similar to size. If not provided, uses size as default. 

    Returns: a Gaussian kernel.
    """
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size_y)))
    return g / g.sum()


gaussian_kernel = gaussian

#=========================================================
# WAVELETS
#=========================================================
"""
Seismic wavelets.

:copyright: 2021 Agile Geoscience
:license: Apache 2.0
"""
def _get_time(duration, dt, sym=None):
    """
    Make a time vector.

    If `sym` is `True`, the time vector will have an odd number of samples,
    and will be symmetric about 0. If it's False, and the number of samples
    is even (e.g. duration = 0.016, dt = 0.004), then 0 will bot be center.
    """
    if sym is None:
        m = "In future releases, the default legacy behaviour will be removed. "
        m += "We recommend setting sym=True. This will be the default in v0.5+."
        warnings.warn(m, category=FutureWarning, stacklevel=2)
        return np.arange(-duration/2, duration/2, dt)
    
    # This business is to avoid some of the issues with `np.arange`:
    # (1) unpredictable length and (2) floating point weirdness, like
    # 1.234e-17 instead of 0. Not using `linspace` because figuring out
    # the length and offset gave me even more of a headache than this.
    n = int(duration / dt)
    odd = n % 2
    k = int(10**-np.floor(np.log10(dt)))
    dti = int(k * dt)  # integer dt
        
    if (odd and sym):
        t = np.arange(n)
    if (not odd and sym):
        t = np.arange(n + 1)
    if (odd and not sym): 
        t = np.arange(n)
    if (not odd and not sym):
        t = np.arange(n) - 1
        
    t -= t[-1] // 2
    
    return dti * t / k


def _generic(func, duration, dt, f, t=None, return_t=False, taper='blackman', sym=None):
    """
    Generic wavelet generator: applies a window to a continuous function.

    Args:
        func (function): The continuous function, taking t, f as arguments.
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Dominant frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis.
        taper (str or function): The window or tapering function to apply.
            To use one of NumPy's functions, pass 'bartlett', 'blackman' (the
            default), 'hamming', or 'hanning'; to apply no tapering, pass
            'none'. To apply your own function, pass a function taking only
            the length of the window and returning the window function.
        sym (bool): If True (default behaviour before v0.5) then the wavelet
            is forced to have an odd number of samples and the central sample
            is at 0 time.

    Returns:
        ndarray. wavelet(s) with centre frequency f sampled on t. If you
            passed `return_t=True` then a tuple of (wavelet, t) is returned.
    """
    if not return_t:
        m = "In future releases, return_t will be True by default."
        warnings.warn(m, FutureWarning)

    f = np.asanyarray(f).reshape(-1, 1)

    # Compute time domain response.
    if t is None:
        t = _get_time(duration, dt, sym=sym)
    else:
        if (duration is not None) or (dt is not None):
            m = "`duration` and `dt` are ignored when `t` is passed."
            warnings.warn(m, UserWarning, stacklevel=2)

    t[t == 0] = 1e-12  # Avoid division by zero.
    f[f == 0] = 1e-12  # Avoid division by zero.

    w = np.squeeze(func(t, f))

    if taper:
        tapers = {
            'bartlett': np.bartlett,
            'blackman': np.blackman,
            'hamming': np.hamming,
            'hanning': np.hanning,
            'none': lambda _: 1,
        }
        taper = tapers.get(taper, taper)
        w *= taper(t.size)

    if return_t:
        Wavelet = namedtuple('Wavelet', ['amplitude', 'time'])
        return Wavelet(w, t)
    else:
        return w


def sinc(duration, dt, f, t=None, return_t=False, taper='blackman', sym=None):
    """
    sinc function centered on t=0, with a dominant frequency of f Hz.

    If you pass a 1D array of frequencies, you get a wavelet bank in return.

    .. plot::
        plt.plot(bruges.filters.sinc(.5, 0.002, 40))

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Dominant frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis.
        taper (str or function): The window or tapering function to apply.
            To use one of NumPy's functions, pass 'bartlett', 'blackman' (the
            default), 'hamming', or 'hanning'; to apply no tapering, pass
            'none'. To apply your own function, pass a function taking only
            the length of the window and returning the window function.

    Returns:
        ndarray. sinc wavelet(s) with centre frequency f sampled on t. If
            you passed `return_t=True` then a tuple of (wavelet, t) is returned.
    """
    def func(t_, f_):
        return np.sin(2*np.pi*f_*t_) / (2*np.pi*f_*t_)

    return _generic(func, duration, dt, f, t, return_t, taper)


def cosine(duration, dt, f, t=None, return_t=False, taper='gaussian', sigma=None, sym=None):
    """
    With the default Gaussian window, equivalent to a 'modified Morlet'
    also sometimes called a 'Gabor' wavelet. The `bruges.filters.gabor`
    function returns a similar shape, but with a higher mean frequancy,
    somewhere between a Ricker and a cosine (pure tone).

    If you pass a 1D array of frequencies, you get a wavelet bank in return.

    .. plot::
        plt.plot(bruges.filters.cosine(.5, 0.002, 40))

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Dominant frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis.
        taper (str or function): The window or tapering function to apply.
            To use one of NumPy's functions, pass 'bartlett', 'blackman' (the
            default), 'hamming', or 'hanning'; to apply no tapering, pass
            'none'. To apply your own function, pass a function taking only
            the length of the window and returning the window function.
        sigma (float): Width of the default Gaussian window, in seconds.
            Defaults to 1/8 of the duration.

    Returns:
        ndarray. sinc wavelet(s) with centre frequency f sampled on t. If
            you passed `return_t=True` then a tuple of (wavelet, t) is returned.
    """
    if sigma is None:
        sigma = duration / 8

    def func(t_, f_):
        return np.cos(2 * np.pi * f_ * t_)

    def taper(length):
        return scipy.signal.gaussian(length, sigma/dt)

    return _generic(func, duration, dt, f, t, return_t, taper)


def gabor(duration, dt, f, t=None, return_t=False, sym=None):
    """
    Generates a Gabor wavelet with a peak frequency f0 at time t.

    https://en.wikipedia.org/wiki/Gabor_wavelet

    If you pass a 1D array of frequencies, you get a wavelet bank in return.

    .. plot::
        plt.plot(bruges.filters.gabor(.5, 0.002, 40))

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Centre frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis.

    Returns:
        ndarray. Gabor wavelet(s) with centre frequency f sampled on t. If
            you passed `return_t=True` then a tuple of (wavelet, t) is returned.
    """
    def func(t_, f_):
        return np.exp(-2 * f_**2 * t_**2) * np.cos(2 * np.pi * f_ * t_)

    return _generic(func, duration, dt, f, t, return_t)


def ricker(duration, dt, f, t=None, return_t=False, sym=None):
    """
    Also known as the mexican hat wavelet, models the function:

    .. math::
        A =  (1 - 2 \pi^2 f^2 t^2) e^{-\pi^2 f^2 t^2}

    If you pass a 1D array of frequencies, you get a wavelet bank in return.

    .. plot::
        plt.plot(bruges.filters.ricker(.5, 0.002, 40))

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Centre frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis.
        sym (bool): If True (default behaviour before v0.5) then the wavelet
            is forced to have an odd number of samples and the central sample
            is at 0 time.

    Returns:
        ndarray. Ricker wavelet(s) with centre frequency f sampled on t. If
            you passed `return_t=True` then a tuple of (wavelet, t) is returned.
    """
    if not return_t:
        m = "In future releases, return_t will be True by default."
        warnings.warn(m, FutureWarning, stacklevel=2)

    f = np.asanyarray(f).reshape(-1, 1)

    if t is None:
        t = _get_time(duration, dt, sym=sym)
    else:
        if (duration is not None) or (dt is not None):
            m = "`duration` and `dt` are ignored when `t` is passed."
            warnings.warn(m, UserWarning, stacklevel=2)

    pft2 = (np.pi * f * t)**2
    w = np.squeeze((1 - (2 * pft2)) * np.exp(-pft2))

    if return_t:
        RickerWavelet = namedtuple('RickerWavelet', ['amplitude', 'time'])
        return RickerWavelet(w, t)
    else:
        return w


def klauder(duration, dt, f,
            autocorrelate=True,
            t=None,
            return_t=False,
            taper='blackman',
            sym=None,
            **kwargs):
    """
    By default, gives the autocorrelation of a linear frequency modulated
    wavelet (sweep). Uses scipy.signal.chirp, adding dimensions as necessary.

    .. plot::
        plt.plot(bruges.filters.klauder(.5, 0.002, [10, 80]))

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): is the sample interval in seconds (usually 0.001, 0.002,
            or 0.004)
        f (array-like): Upper and lower frequencies. Any sequence like (f1, f2).
            A list of lists will create a wavelet bank.
        autocorrelate (bool): Whether to autocorrelate the sweep(s) to create
            a wavelet. Default is `True`.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis.
        taper (str or function): The window or tapering function to apply.
            To use one of NumPy's functions, pass 'bartlett', 'blackman' (the
            default), 'hamming', or 'hanning'; to apply no tapering, pass
            'none'. To apply your own function, pass a function taking only
            the length of the window and returning the window function.
        sym (bool): If True (default behaviour before v0.5) then the wavelet
            is forced to have an odd number of samples and the central sample
            is at 0 time.
        **kwargs: Further arguments are passed to scipy.signal.chirp. They are
            `method` ('linear','quadratic','logarithmic'), `phi` (phase offset
            in degrees), and `vertex_zero`.

    Returns:
        ndarray: The waveform. If you passed `return_t=True` then a tuple of
            (wavelet, t) is returned.
    """
    if not return_t:
        m = "In future releases, return_t will be True by default."
        warnings.warn(m, FutureWarning, stacklevel=2)

    if t is None:
        t = _get_time(duration, dt, sym=sym)
    else:
        if (duration is not None) or (dt is not None):
            m = "`duration` and `dt` are ignored when `t` is passed. "
            m += "Pass None to suppress this warning."
            warnings.warn(m, UserWarning, stacklevel=2)

    t0, t1 = -duration/2, duration/2

    f = np.asanyarray(f).reshape(-1, 1)
    f1, f2 = f

    c = [scipy.signal.chirp(t, f1_+(f2_-f1_)/2., t1, f2_, **kwargs)
         for f1_, f2_
         in zip(f1, f2)]

    if autocorrelate:
        w = [np.correlate(c_, c_, mode='same') for c_ in c]

    w = np.squeeze(w) / np.amax(w)

    if taper:
        funcs = {
            'bartlett': np.bartlett,
            'blackman': np.blackman,
            'hamming': np.hamming,
            'hanning': np.hanning,
            'none': lambda x: x,
        }
        func = funcs.get(taper, taper)
        w *= func(t.size)

    if return_t:
        Sweep = namedtuple('Sweep', ['amplitude', 'time'])
        return Sweep(w, t)
    else:
        return w


sweep = klauder


def ormsby(duration, dt, f, t=None, return_t=False, sym=None):
    """
    The Ormsby wavelet requires four frequencies which together define a
    trapezoid shape in the spectrum. The Ormsby wavelet has several sidelobes,
    unlike Ricker wavelets.

    .. plot::
        plt.plot(bruges.filters.ormsby(.5, 0.002, [5, 10, 40, 80]))

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (usually 0.001, 0.002,
            or 0.004).
        f (array-like): Sequence of form (f1, f2, f3, f4), or list of lists of
            frequencies, which will return a 2D wavelet bank.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis.
        sym (bool): If True (default behaviour before v0.5) then the wavelet
            is forced to have an odd number of samples and the central sample
            is at 0 time.

    Returns:
        ndarray: A vector containing the Ormsby wavelet, or a bank of them. If
            you passed `return_t=True` then a tuple of (wavelet, t) is returned.

    """
    if not return_t:
        m = "In future releases, return_t will be True by default."
        warnings.warn(m, FutureWarning, stacklevel=2)

    f = np.asanyarray(f).reshape(-1, 1)

    try:
        f1, f2, f3, f4 = f
    except ValueError:
        raise ValueError("The last dimension of the frequency array must be of size 4.")

    def numerator(f, t):
        return (np.sinc(f * t)**2) * ((np.pi * f) ** 2)

    pf43 = (np.pi * f4) - (np.pi * f3)
    pf21 = (np.pi * f2) - (np.pi * f1)

    if t is None:
        t = _get_time(duration, dt, sym=sym)
    else:
        if (duration is not None) or (dt is not None):
            m = "`duration` and `dt` are ignored when `t` is passed."
            warnings.warn(m, UserWarning, stacklevel=2)

    w = ((numerator(f4, t)/pf43) - (numerator(f3, t)/pf43) -
         (numerator(f2, t)/pf21) + (numerator(f1, t)/pf21))

    w = np.squeeze(w) / np.amax(w)

    if return_t:
        OrmsbyWavelet = namedtuple('OrmsbyWavelet', ['amplitude', 'time'])
        return OrmsbyWavelet(w, t)
    else:
        return w


def ormsby_fft(duration, dt, f, P=(0, 0), return_t=True, sym=True):
    """
    Non-white Ormsby, with arbitary amplitudes.
    
    Can use as many points as you like. The power of f1 and f4 is assumed to be 0,
    so you only need to provide p2 and p3 (the corners). (You can actually provide
    as many f points as you like, as long as there are n - 2 matching p points.)

    .. plot::
        plt.plot(bruges.filters.ormsby(.5, 0.002, [5, 10, 40, 80]))

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (usually 0.001, 0.002,
            or 0.004).
        f (array-like): Sequence of form (f1, f2, f3, f4), or list of lists of
            frequencies, which will return a 2D wavelet bank.
        P (tuple): The power of the f2 and f3 frequencies, in relative dB.
            (The magnitudes of f1 and f4 are assumed to be -∞ dB, i.e. a
            magnitude of 0.) The default power values of (0, 0) results in a
            trapezoidal spectrum and a conventional Ormsby wavelet. Pass, e.g.
            (0, -15) for a 'pink' wavelet, with more energy in the lower
            frequencies.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis.
        sym (bool): If True (default behaviour before v0.5) then the wavelet
            is forced to have an odd number of samples and the central sample
            is at 0 time.

    Returns:
        ndarray: A vector containing the Ormsby wavelet, or a bank of them. If
            you passed `return_t=True` then a tuple of (wavelet, t) is returned.
    """
    fs = 1 / dt
    fN = fs // 2
    n = int(duration / dt)
    a = map(lambda p: 10**(p/20), P)

    # Linear interpolation of points.
    x  = np.linspace(0, int(fN), int(10*n))
    xp = [  0.] + list(f) +  [fN]
    fp = [0., 0.] + list(a) + [0., 0.]
    W = np.interp(x, xp, fp)

    # Compute inverse FFT.
    w_ = np.fft.fftshift(np.fft.irfft(W))
    L = int(w_.size // 2)
    normalize = lambda d: d / np.max(abs(d))
    w = normalize(w_[L-n//2:L+n//2+sym])
    t = _get_time(duration, dt, sym=sym)

    if return_t:
        OrmsbyWavelet = namedtuple('OrmsbyWavelet', ['amplitude', 'time'])
        return OrmsbyWavelet(w, t)
    else:
        return w


def berlage(duration, dt, f, n=2, alpha=180, phi=-np.pi/2, t=None, return_t=False, sym=None):
    """
    Generates a Berlage wavelet with a peak frequency f. Implements

    .. math::

        w(t) = AH(t) t^n \mathrm{e}^{-\alpha t} \cos(2 \pi f_0 t + \phi_0)

    as described in Aldridge, DF (1990), The Berlage wavelet, GEOPHYSICS
    55 (11), p 1508-1511. Berlage wavelets are causal, minimum phase and
    useful for modeling marine airgun sources.

    If you pass a 1D array of frequencies, you get a wavelet bank in return.

    .. plot::
        plt.plot(bruges.filters.berlage(0.5, 0.002, 40))

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Centre frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        n (float): The time exponent; non-negative and real.
        alpha(float): The exponential decay factor; non-negative and real.
        phi (float): The phase.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis.
        sym (bool): If True (default behaviour before v0.5) then the wavelet
            is forced to have an odd number of samples and the central sample
            is at 0 time.

    Returns:
        ndarray. Berlage wavelet(s) with centre frequency f sampled on t. If
            you passed `return_t=True` then a tuple of (wavelet, t) is returned.
    """
    if not return_t:
        m = "In future releases, return_t will be True by default."
        warnings.warn(m, FutureWarning, stacklevel=2)

    f = np.asanyarray(f).reshape(-1, 1)
    if t is None:
        t = _get_time(duration, dt, sym=sym)
    else:
        if (duration is not None) or (dt is not None):
            m = "`duration` and `dt` are ignored when `t` is passed."
            warnings.warn(m, UserWarning, stacklevel=2)


    H = np.heaviside(t, 0)
    w = H * t**n * np.exp(-alpha * t) * np.cos(2 * np.pi * f * t + phi)

    w = np.squeeze(w) / np.max(np.abs(w))

    if return_t:
        BerlageWavelet = namedtuple('BerlageWavelet', ['amplitude', 'time'])
        return BerlageWavelet(w, t)
    else:
        return w


def generalized(duration, dt, f, u=2, t=None, return_t=False, imag=False, sym=None):
    """
    Wang's generalized wavelet, of which the Ricker is a special case where
    u = 2. The parameter u is the order of the time-domain derivative, which
    can be a fractional derivative.

    As given by Wang (2015), Generalized seismic wavelets. GJI 203, p 1172-78.
    DOI: https://doi.org/10.1093/gji/ggv346. I am using the (more accurate)
    frequency domain method (eq 4 in that paper).

    .. plot::
        plt.plot(bruges.filters.generalized(.5, 0.002, 40, u=1.0))

    Args:
        duration (float): The length of the wavelet, in s.
        dt (float): The time sample interval in s.
        f (float or array-like): The frequency or frequencies, in Hertz.
        u (float or array-like): The fractional derivative parameter u.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        return_t (bool): Whether to return the time basis array.
        center (bool): Whether to center the wavelet on time 0.
        imag (bool): Whether to return the imaginary component as well.
        sym (bool): If True (default behaviour before v0.5) then the wavelet
            is forced to have an odd number of samples and the central sample
            is at 0 time.

    Returns:
        ndarray. If f and u are floats, the resulting wavelet has duration/dt
            = A samples. If you give f as an array of length M and u as an
            array of length N, then the resulting wavelet bank will have shape
            (M, N, A). If f or u are floats, their size will be 1, and they
            will be squeezed out: the bank is always squeezed to its minimum
            number of dimensions. If you passed `return_t=True` then a tuple
            of (wavelet, t) is returned.
    """
    if not return_t:
        m = "In future releases, return_t will be True by default."
        warnings.warn(m, FutureWarning, stacklevel=2)

    # Make sure we can do banks.
    f = np.asanyarray(f).reshape(-1, 1)
    u = np.asanyarray(u).reshape(-1, 1, 1)

    # Compute time domain response.
    if t is None:
        t = _get_time(duration, dt, sym=sym)
    else:
        if (duration is not None) or (dt is not None):
            m = "`duration` and `dt` are ignored when `t` is passed."
            warnings.warn(m, UserWarning, stacklevel=2)
        dt = t[1] - t[0]
        duration = len(t) * dt

    # Basics.
    om0 = f * 2 * np.pi
    u2 = u / 2
    df = 1 / duration
    nyquist = (1 / dt) / 2
    nf = 1 + nyquist / df
    t0 = duration / 2
    om = 2 * np.pi * np.arange(0, nyquist, df)

    # Compute the spectrum from Wang's eq 4.
    exp1 = np.exp((-om**2 / om0**2) + u2)
    exp2 = np.exp(-1j*om*t0 + 1j*np.pi * (1 + u2))
    W = (u2**(-u2)) * (om**u / om0**u) * exp1 * exp2

    w = np.fft.ifft(W, t.size)
    if not imag:
        w = w.real

    # At this point the wavelet bank has the shape (u, f, a),
    # where u is the size of u, f is the size of f, and a is
    # the number of amplitude samples we generated.
    w_max = np.max(np.abs(w), axis=-1)[:, :, None]
    w = np.squeeze(w / w_max)

    if return_t:
        GeneralizedWavelet = namedtuple('GeneralizedWavelet', ['amplitude', 'time'])
        return GeneralizedWavelet(w, t)
    else:
        return w


@deprecated('bruges.filters.wavelets.rotate_phase() is deprecated. Please use bruges.filters.rotate_phase() instead.')
def rotate_phase(w, phi, degrees=False):
    """
    Performs a phase rotation of wavelet or wavelet bank using:

    .. math::

        A = w(t)\cos(\phi) - h(t)\sin(\phi)

    where w(t) is the wavelet and h(t) is its Hilbert transform.

    The analytic signal can be written in the form S(t) = A(t)exp(j*theta(t))
    where A(t) = magnitude(hilbert(w(t))) and theta(t) = angle(hilbert(w(t))
    then a constant phase rotation phi would produce the analytic signal
    S(t) = A(t)exp(j*(theta(t) + phi)). To get the non analytic signal
    we take real(S(t)) == A(t)cos(theta(t) + phi)
    == A(t)(cos(theta(t))cos(phi) - sin(theta(t))sin(phi)) <= trig identity
    == w(t)cos(phi) - h(t)sin(phi)

    Args:
        w (ndarray): The wavelet vector, can be a 2D wavelet bank.
        phi (float): The phase rotation angle (in radians) to apply.
        degrees (bool): If phi is in degrees not radians.

    Returns:
        The phase rotated signal (or bank of signals).
    """
    if degrees:
        phi = phi * np.pi / 180.0
    a = scipy.signal.hilbert(w, axis=0)
    w = np.real(a) * np.cos(phi)  -  np.imag(a) * np.sin(phi)
    return w

#=========================================================
# CONVOLVE
#=========================================================
"""
Convolution in n-dimensions.

:copyright: 2019 Agile Geoscience
:license: Apache 2.0
"""
def convolve(reflectivity, wavelet):
    """
    Convolve n-dimensional reflectivity with a 1D wavelet or 2D wavelet bank.
    
    Args
    reflectivity (ndarray): The reflectivity trace, or 2D section, or volume.
    wavelet (ndarray): The wavelet, must be 1D function or a 2D wavelet 'bank'.
        If a wavelet bank, time should be on the last axis.
    """
    # Compute the target shape of the final synthetic.
    outshape = wavelet.shape[:-1] + reflectivity.shape

    # Force wavelet and reflectivity to both be 2D.
    bank = np.atleast_2d(wavelet)   
    reflectivity_2d = reflectivity.reshape((-1, reflectivity.shape[-1]))

    # Compute synthetic, which will always be 3D.
    syn = np.array([apply_along_axis(np.convolve, reflectivity_2d, w, mode='same') for w in bank])

    return syn.reshape(outshape)

#=========================================================
# ANISOTROPY DIFF
#=========================================================
"""
:copyright: Alistair Muldal
:license: Unknown, shared on StackOverflow and Pastebin

Reference:
P. Perona and J. Malik.
Scale-space and edge detection using ansotropic diffusion.
IEEE Transactions on Pattern Analysis and Machine Intelligence,
12(7):629-639, July 1990.
<http://www.cs.berkeley.edu/~malik/papers/MP-aniso.pdf>

Original MATLAB code by Peter Kovesi
School of Computer Science & Software Engineering
The University of Western Australia
pk @ csse uwa edu au
<http://www.csse.uwa.edu.au>

Translated to Python and optimised by Alistair Muldal
Department of Pharmacology
University of Oxford
<alistair.muldal@pharm.ox.ac.uk>

June 2000  original version.
March 2002 corrected diffusion eqn No 2.
July 2012 translated to Python
"""
def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1., 1.), option=1):
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence
    diffusion across step edges.  A large value reduces the influence of
    intensity gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between
    adjacent pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast
    ones.
    Diffusion equation 2 favours wide regions over smaller ones.
    """

    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        m = "Only grayscale images allowed, converting to 2D matrix"
        warnings.warn(m)
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for ii in xrange(niter):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
                gS = np.exp(-(deltaS/kappa)**2.)/step[0]
                gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
                gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
                gE = 1./(1.+(deltaE/kappa)**2.)/step[1]

        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma*(NS+EW)

    return imgout


def anisodiff3(stack, niter=1, kappa=50, gamma=0.1, step=(1., 1., 1.), option=1):
    """
    3D Anisotropic diffusion.

    Usage:
    stackout = anisodiff(stack, niter, kappa, gamma, option)

    Arguments:
        stack  - input stack
        niter  - number of iterations
        kappa  - conduction coefficient 20-100 ?
        gamma  - max value of .25 for stability
        step   - tuple, the distance between adjacent pixels in (z,y,x)
        option - 1 Perona Malik diffusion equation No 1
                 2 Perona Malik diffusion equation No 2

    Returns:
        stackout   - diffused stack.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence
    diffusion across step edges. A large value reduces the influence of
    intensity gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between
    adjacent pixels differs in the x,y and/or z axes

    Diffusion equation 1 favours high contrast edges over low contrast
    ones.
    Diffusion equation 2 favours wide regions over smaller ones.
    """

    # ...you could always diffuse each color channel independently if you
    # really want
    if stack.ndim == 4:
        m = "Only grayscale stacks allowed, converting to 3D matrix"
        warnings.warn(m)
        stack = stack.mean(3)

    # initialize output array
    stack = stack.astype('float32')
    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    for ii in range(niter):

        # calculate the diffs
        deltaD[:-1, :, :] = np.diff(stackout, axis=0)
        deltaS[:, :-1, :] = np.diff(stackout, axis=1)
        deltaE[:, :, :-1] = np.diff(stackout, axis=2)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
                gD = np.exp(-(deltaD/kappa)**2.)/step[0]
                gS = np.exp(-(deltaS/kappa)**2.)/step[1]
                gE = np.exp(-(deltaE/kappa)**2.)/step[2]
        elif option == 2:
                gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
                gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
                gE = 1./(1.+(deltaE/kappa)**2.)/step[2]

        # Update matrices.
        D = gD*deltaD
        E = gE*deltaE
        S = gS*deltaS

        # Subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. Don't ask questions. Just do it. Trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:, :, :] -= D[:-1, :, :]
        NS[:, 1:, :] -= S[:, :-1, :]
        EW[:, :, 1:] -= E[:, :, :-1]

        # update the image
        stackout += gamma*(UD+NS+EW)

    return stackout


#================================================================================================
# MODELS
#================================================================================================
# PANEL
#=========================================================
import numpy as np
from scipy.ndimage import zoom
from scipy.interpolate import interp1d


def reconcile(*arrays, order=0):
    """
    Make sure 1D arrays are the same length. If not, stretch them to match
    the longest.

    Args:
        arrays (ndarray): The input arrays.
        order (int): The order of the interpolation, passed to
            scipy.ndimage.zoom. Suggestion: 0 for integers and 1 for floats.

    Returns:
        tuple of ndarrays. The reconciled arrays — all of them are now the
            same length.

    Example:
        >>> a = np.array([2, 6, 7, 7, 3])
        >>> b = np.array([3, 7, 3])
        >>> reconcile(a, b, order=0)
        (array([2, 6, 7, 7, 3]), array([3, 7, 7, 3, 3]))
    """
    maxl = max(len(arr) for arr in arrays)
    out = []
    for arr in arrays:
        if len(arr) < maxl:
            out.append(zoom(arr, zoom=maxl/len(arr), order=order))
        else:
            out.append(arr)
    return tuple(out)


def interpolate(*arrays, num=50, dists=None, kind='linear'):
    """
    Linear interpolation between 1D arrays of the same length.

    Args:
        arrays (ndarray): The 1D arrays to interpolate. All must be the same
            length. You can use the `reconcile()` function to produce them.
        num (int): The number of steps to take, so will be the width (number
            of cols) of the output array.
        dists (array-like): A list or tuple or array of the distances (any
            units) between the arrays in the real world.
        kind (str): Will be passed to scipy.interpolate.interp1d, which does
            the lateral interpolation between samples.

    Returns:
        ndarray. The result, with `num` columns. The number of rows is the
            same as the number of samples in the input arrays.

    Example:
        >>> a = np.array([2, 6, 7, 7, 3])
        >>> b = np.array([3, 7, 7, 3, 3])
        >>> interp = interpolate(a, b, num=10)
        >>> interp.shape
        (5, 10)
    """
    intervals = len(arrays) - 1
    if dists is None:
        dists = intervals * [num / intervals]
    x = np.hstack([[0], np.cumsum(dists)])
    f = interp1d(x, np.stack(arrays), axis=0, kind=kind)
    return f(np.linspace(x[0], x[-1], num=num)).T


def unreconcile(arr, sizes, dists=None, order=0):
    """
    Opposite of reconcile. Restores the various profiles (the reference arrays,
    e.g. wells) to their original lengths.

    Args:
        sizes (int): The relative lengths of the profiles in the array.
            Default returns the input array.
        dists (array-like): The relative distances between the profiles in
            the array. Sum used to calculate the output width in pixels if
            the width argument is None. If not given, the distances are
            assumed to be equal.
        order (int): The order of the spline interpolation, from 0 to 3. The
            default is 0, which gives nearest neighbour interpolation. 1 gives
            linear interpolation, etc. Use 0 for ints and 1-3 for floats.

    Returns:
        ndarray. The resulting ndarray. The array contains NaNs where there
            is no interpolated data.
    """
    if np.all(sizes[0] == np.array(sizes)):
        # Nothing to do.
        return arr

    intervals = len(sizes) - 1

    if dists is None:
        eq = arr.shape[-1] // intervals
        dists = [eq] * intervals
    assert len(dists) == intervals

    maxlen = int(np.ceil(max(sizes) * arr.shape[0]))

    dist_ = np.cumsum(dists)
    idx = arr.shape[-1] * dist_ / max(dist_)
    chunks = np.split(arr, idx[:-1].astype(int), axis=-1)

    zoomed = []
    for left, right, chunk in zip(sizes[:-1], sizes[1:], chunks):
        zooms = np.linspace(left, right, chunk.shape[-1]+1)
        for z, col in zip(zooms, chunk.T):
            new_ = zoom(col, zoom=z, order=order, mode='nearest')
            pad_width = maxlen - new_.size
            new = np.pad(new_,
                         pad_width=(0, pad_width),
                         mode='constant',
                         constant_values=np.nan,
                         )
            zoomed.append(new)

    return np.array(zoomed).T


def panel(*arrays, num=50, dists=None, order=0, kind='linear'):
    """
    Interpolate an arbitrary collection of 1D arrays.

    Args:
        num (int): The number of steps to take, so will be the width (number
            of cols) of the output array.
        dists (array-like): The relative distances between the profiles in
            the array. Sum used to calculate the output width in pixels if
            the width argument is None. If not given, the distances are
            assumed to be equal.
        order (int): The order of the interpolation, passed to
            scipy.ndimage.zoom. Suggestion: 0 for integers and 1 for floats.
        kind (str): Will be passed to scipy.interpolate.interp1d, which does
            the lateral interpolation between samples.

    Returns:
        ndarray. The interpolated panel. Contains NaNs if sizes are
            non-uniform.
    """
    sizes = np.array([len(x) for x in arrays])
    sizes = sizes / np.max(sizes)
    rec = reconcile(*arrays)
    interp = interpolate(*rec, num=num, dists=dists, kind=kind)
    panel = unreconcile(interp, sizes=sizes, dists=dists, order=order)
    return panel

#=========================================================
# WEDGE
#=========================================================
def pad_func(before, after):
    """
    Padding function. Operates on vector *in place*, per the np.pad
    documentation.
    """
    def pad_with(x, pad_width, iaxis, kwargs):
        x[:pad_width[0]] = before[-pad_width[0]:]
        x[-pad_width[1]:] = after[:pad_width[1]]
        return
    return pad_with


def get_strat(strat,
              thickness,
              kind='nearest',
              position=1,
              wedge=None,
              zoom_mode='nearest'
              ):
    """
    Take a 'stratigraphy' (either an int, a tuple of ints, or a list-like of
    floats) and expand or compress it to the required thickness.

    `kind` can be 'nearest', 'linear', 'quadratic', or 'cubic'.
    """
    orders = {'nearest': 0, 'linear': 1, 'quadratic': 2, 'cubic': 3}
    order = orders.get(kind, 0)

    if isinstance(strat, int) and (order == 0):
        out = np.repeat([strat], thickness)
    elif isinstance(strat, float) and (order == 0):
        out = np.repeat([strat], thickness)
    elif isinstance(strat, tuple) and (order == 0):
        out = np.repeat(strat, int(round(thickness/len(strat))))
    else:
        if position == 0:
            wedge_zoom = wedge[1]/len(wedge[0])
            strat = strat[-int(thickness/wedge_zoom):]
        elif position == -1:
            wedge_zoom = wedge[1]/len(wedge[0])
            strat = strat[:int(thickness/wedge_zoom)]
        zoom = thickness / len(strat)
        out = sn.zoom(strat, zoom=zoom, order=order, mode=zoom_mode)

    # Guarantee correct length by adjusting bottom layer.
    missing = int(np.ceil(thickness - out.size))
    if out.size > 0 and missing > 0:
        out = np.pad(out, [0, missing], mode='edge')
    elif out.size > 0 and missing < 0:
        out = out[:missing]

    return out


def get_conforming(strat, thickness, conformance):
    """
    Function to deal with top and bottom conforming wedges.
    """
    thickness = int(np.ceil(thickness))
    if thickness == 0:
        return np.array([])
    if strat.size == thickness:
        return strat
    elif strat.size > thickness:
        if conformance == 'top':
            return strat[:thickness]
        else:
            return strat[-thickness:]
    else:
        if conformance == 'top':
            return np.pad(strat, [0, thickness-strat.size], mode='wrap')
        else:
            return np.pad(strat, [thickness-strat.size, 0], mode='wrap')
    return


def get_subwedges(target, breadth):
    """
    For a binary target (the reference trace of a wedge),
    create the range of net:gross subwedges. We do this
    with binary morphologies in the following way:

    - Erode the 'secondary' component (whatever appears
      second in the target array) one step at a time,
      until there is nothing left and the resulting
      trace contains only the primary.
    - Dilate the secondary component until there is
      nothing left of the primary and the resulting trace
      contains only the secondary.
    - Arrange the traces in order, starting with all
      primary, and ending in all secondary. The target
      trace will be somewhere in between, but not
      necessarily in the middle.

    Returns a 2D array with one target wedge trace per
    section.

    Args:
        target (array): A 1D array length N, the 'reference'
            trace for the wedge. The reference trace has
            thickness '1' in the wedge model. This trace must
            be 'binary' — i.e. it must contain exactly 2
            unique values.
        breadth (int): How many new reference traces should
            be in the output.

    Returns:
        tuple (ndarray, ndarray, int). The ndarray has shape
            N x breadth. It represents one target wedge trace
            per section in 'breadth'. The integer is the
            position of the target trace in the ndarray's
            second dimension.
    """
    try:
        components = a, b = np.unique(target)
    except ValueError:
        raise ValueError("Must be a binary (2-component) wedge.")

    out = [target]

    temp = target.copy()
    while b in temp:
        ero = morph.binary_erosion(temp == b)
        temp = components[ero.astype(int)]
        out.append(temp)

    out = out[::-1]
    ref = len(out)

    temp = target.copy()
    while a in temp:
        dil = morph.binary_dilation(temp == b)
        temp = components[dil.astype(int)]
        out.append(temp)

    arr_ = np.array(out).T
    h, w = arr_.shape
    arr = sn.zoom(arr_, zoom=(1, breadth/w), mode='nearest')

    ng = np.divide(np.sum(arr == a, axis=0), target.size)

    return arr, ng, ref * breadth / w


def wedge(depth=(30, 40, 30),
          width=(10, 80, 10),
          breadth=None,
          strat=(0, 1, 2),
          thickness=(0.0, 1.0),
          mode='linear',
          conformance='both',
          ):
    """
    Generate a wedge model.

    Args:
        depth (int or tuple): The vertical size of the model. If a 3-tuple,
            then each element corresponds to a layer. If an integer, then each
            layer of the model will be 1/3 of the thickness. Note that if the
            'right' wedge thickness is more than 1, then the total thickness
            will be greater than this value.
        width (int or tuple): The width of the model. If a 3-tuple, then each
            element corresponds to a 'zone' (left, middle, right). If an
            integer, then the zones will be 10%, 80% and 10% of the width,
            respectively.
        breadth (None or int): Not implemented. Raises an error.
        strat (tuple): Stratigraphy above, in, and below the wedge. This is the
            'reference' stratigraphy. If you give integers, you get 'solid'
            layers containing those numbers. If you give arrays, you will get
            layers of those arrays, expanded or squeezed into the layer
            thicknesses implied in `depth`.
        thickness (tuple): The wedge thickness on the left and on the right.
            Default is (0.0, 1.0) so the wedge will be thickness 0 on the left
            and the wedge thickness specified in the depth argument on the
            right. If the thickness are equal, you'll have a flat, layer-cake
            model.
        mode (str or function): What kind of interpolation to use. Default:
            'linear'. Other options are 'sigmoid', which makes a clinoform-like
            body, 'root', which makes a steep-sided bowl shape like the edge of
            a channel, and 'power', which makes a less steep-sided bowl shape.
            If you pass a function, give it an API like np.linspace: f(start,
            stop, num), where start is the left thickness, stop is the right
            thickness, and num is the width of the middle (wedge) 'zone'.
        conformance (str): 'top', 'bottom', or 'both' (the default). How you
            want the layers inside the wedge to behave. For top and bottom
            conformance, if the layer needs to be thicker than the reference.

    Returns:
        namedtuple[ndarray, ndarray, ndarray, int]: A tuple containing the
            2D wedge model, the top 'horizon', the base 'horizon', and the
            position at which the wedge has thickness 1 (i.e. is the thickness
            specfied by the middle layer depth and/or strat).
    """
    # Decide if binary (2 rocks in the wedge).
    if np.unique(strat[1]).size == 2:
        binary = True
    else:
        binary = False

    if breadth is None or breadth == 0:
        breadth = 1

    # Allow wedge to be thin-thick or thick-thin.
    left, right = thickness
    if left > right:
        left, right = right, left
        flip = True
    else:
        flip = False

    # Get all layers thicknesses.
    if isinstance(depth, int):
        L1, L2, L3 = 3 * [depth//3]  # Sizes if depth is just a number.
        L3 += 1
    else:
        L1, L2, L3 = map(int, depth)
    L3 += int(right * L2)  # Adjust bottom layer.

    # Get all zone widths.
    if isinstance(width, int):
        Z2 = round(0.8 * width)
        Z1 = round(max(1, width/10))
        Z3 = width - Z2 - Z1
        width = int(Z1), int(Z2), int(Z3)
    else:
        Z1, Z2, Z3 = width  # Z1 and Z3 are the bookends.

    # Deal with different interpolation patterns.
    modes = {
        'linear': np.linspace,
        'clinoform': sigmoid,
        'sigmoid': sigmoid,
        'root': root,
        'power': power,
    }
    zooms = modes.get(mode, mode)(left, right, Z2)

    # Get the reference stratigraphy in each layer.
    # The 'well log' case is tricky, because layer1 and layer3
    # need to know about the amount of zoom on the wedge layer.
    # There must be an easier way to do this.
    layer1 = get_strat(strat[0], L1, position=0, wedge=(strat[1], L2))
    layer2_ = get_strat(strat[1], L2, position=1)
    layer3 = get_strat(strat[2], L3, position=-1, wedge=(strat[1], L2))

    # Deal with width. We discard the reference breadth.
    if binary and (breadth >= 2):
        layer2s, _n2g, _ = get_subwedges(layer2_, breadth=breadth)
        layer2s = layer2s.T
    else:
        layer2s, _ = [layer2_], None

    # Make the padding function.
    padder = pad_func(layer1, layer3)

    # For everything in breadth:
    model = []
    for layer2 in layer2s:
        # Collect wedge pieces, then pad top & bottom, then stack,
        # then pad left & right.
        if conformance in ['top', 'bottom', 'base']:
            wedges = [get_conforming(layer2, z*L2, conformance) for z in zooms]
        else:
            wedges = [get_strat(layer2, thickness=z*L2) for z in zooms]
        padded = [np.pad(w, [L1, L3-w.size], mode=padder) for w in wedges]
        wedge = np.pad(np.stack(padded), [[Z1, Z3], [0, 0]], mode='edge')
        model.append(wedge.T)
    model = np.array(model)

    # Make the top and base 'horizons'.
    top = np.repeat((np.ones(np.sum(width)) * L1)[:, None], breadth, axis=-1)
    base_ = np.pad(L1 + zooms * L2, [Z1, Z3], mode='edge')
    base = np.repeat(base_[:, None], breadth, axis=-1)

    # Calculate the reference profile ('well' position).
    if left <= 1 <= right:
        ref = Z1 + np.argmin(np.abs(zooms-1))
    elif left == right == 1:
        ref = Z1 + Z2//2
    else:
        ref = -1

    # Flip if required.
    if flip:
        model = np.flip(model, axis=2)
        base = base[::-1]
        ref = sum(width) - ref

    # Move the 'breadth' dim to last.
    if model.shape[0] > 1:
        model = np.moveaxis(model, 0, -1)

    # Build and return output.
    Wedge = namedtuple('Wedge', ['wedge', 'top', 'base', 'reference'])
    return Wedge(np.squeeze(model),
                 np.squeeze(top),
                 np.squeeze(base),
                 ref
                 )



#================================================================================================
# VARIOUS TRANSFORMS
#================================================================================================
def ft_m(x):
    return x / 0.3048


def m_ft(x):
    return x * 0.3048

#=========================================================
# VELOCITIES
#=========================================================
def v_rms(v, depth=None, time=None):
    """
    Cumulative RMS mean of a velocity log. You must provide either
    a depth or a time basis for the log.

    Args:
        v (ndarray): The velocity log.
        depth (ndarray): The depth values corresponding to the log.
        time (ndarray): The time values corresponding to the log.

    Returns:
        ndarray: The V_rms log.
    """
    if (depth is None) and (time is None):
        raise TypeError("You must provide a depth or time array")

    if depth is None:
        return np.sqrt(np.cumsum(v**2 / time) / np.cumsum(time))
    else:
        return np.sqrt(np.cumsum(depth * v) / np.cumsum(depth / v))


def v_avg(v, depth=None, time=None):
    """
    Cumulative average of a velocity log. You must provide either
    a depth or a time basis for the log.

    Args:
        v (ndarray): The velocity log.
        depth (ndarray): The depth values corresponding to the log.
        time (ndarray): The time values corresponding to the log.

    Returns:
        ndarray: The V_avg log.
    """
    if (depth is None) and (time is None):
        raise TypeError("You must provide a depth or time array")

    if depth is None:
        return np.cumsum(v * time) / np.cumsum(time)
    else:
        return np.cumsum(depth) / np.cumsum(depth / v)


def v_bac(v, rho, depth):
    """
    Cumulative Backus average of a velocity log. You must provide
    either a depth or a time basis for the log.

    For a non-cumulative version that can also provide sclaing for the
    V_s log, as well as quality factor, see bruges.anisotropy.backus.

    Args:
        v (ndarray): The velocity log.
        rho (ndarray): The density log.
        depth (ndarray): The depth values corresponding to the logs.

    Returns:
        ndarray: The V_bac log.
    """
    num = np.cumsum(depth**2)
    den = np.cumsum(rho * depth) * np.cumsum(depth/(v**2 * rho))
    return np.sqrt(num / den)

def reflection_time(t0, x, vnmo):
    """
    Calculate the travel-time of a reflected wave. Doesn't consider
    refractions or changes in velocity.

    The units must be consistent. E.g., if t0 is seconds and
    x is metres, vnmo must be m/s.

    Args:
        t0 (float): The 0-offset (normal incidence) travel-time.
        x (float): The offset of the receiver.
        vnmo (float): The NMO velocity.

    Returns:
        t (float): The reflection travel-time.
    """
    t = np.sqrt(t0**2 + x**2/vnmo**2)
    return t


def sample_trace(trace, time, dt):
    """
    Sample an amplitude at a given time using interpolation.

    Args:
        trace (1D array): Array containing the amplitudes of a single trace.
        time (float): The time at which I want to sample the amplitude.
        dt (float): The sampling interval.

    Returns:
        amplitude (float or None): The interpolated amplitude. Will be None
        if *time* is beyond the end of the trace or if there are fewer than
        two points between *time* and the end.
    """
    # Use floor to get the index that is right before our desired time.
    before = int(np.floor(time/dt))
    N = trace.size

    # Use the 4 samples around time to interpolate
    samples = np.arange(before - 1, before + 3)
    if any(samples < 0) or any(samples >= N):
        amplitude = None
    else:
        times = dt * samples
        amps = trace[samples]
        interpolator = CubicSpline(times, amps)
        amplitude = interpolator(time)
    return amplitude


def nmo_correction(cmp, dt, offsets, velocities):
    """
    Performs NMO correction on the given CMP.

    The units must be consistent. E.g., if dt is seconds and
    offsets is meters, velocities must be m/s.

    Args:
        cmp (ndarray): The 2D array CMP gather that we want to correct.
        dt (float): The sampling interval.
        offsets (ndarray): A 1D array with the offset of each trace in the CMP.
        velocities (ndarray): A 1D array with the NMO velocity for each time.
            Should have the same number of elements as the CMP has samples.

    Returns:
        ndrray: The NMO corrected gather.

    """
    nmo = np.zeros_like(cmp)
    nsamples = cmp.shape[0]
    times = np.arange(0, nsamples*dt, dt)
    for i, t0 in enumerate(times):
        for j, x in enumerate(offsets):
            t = reflection_time(t0, x, velocities[i])
            amplitude = sample_trace(cmp[:, j], t, dt)
            # If the time t is outside of the CMP time range,
            # amplitude will be None.
            if amplitude is not None:
                nmo[i, j] = amplitude
    return nmo

def __convert(data, vmodel, interval, interval_new, scale, mode, return_basis=False):
    """
    Generic function for converting between scales. Use either
    time to depth or depth to time.

        Args:
        data (ndarray): The data to convert, will work with a 1 or 2D numpy
            numpy array. array(samples,traces).
        vmodel (ndarray): P-wave interval velocity model that corresponds to
            the data. Must be the same shape as data.
       interval (float): The sample interval of the input data [s] or [m].
       interval_new (float): The sample interval of the output data [m] or [s].
       mode (str): What kind of interpolation to use, defaults to 'nearest'.
        return_basis (bool): Whether to also return the new time basis.

    Returns
        ndarray: The data resampled in the depth domain.
    """
    data = np.array(data)

    if np.ndim(data) == 1:
        ntraces = 1
        nsamps = data.size
        if np.size(interval) == 1:
            basis = np.arange(nsamps)*interval
        else:
            basis = interval
        v_avg = np.cumsum(vmodel) / (np.arange(nsamps) + 1)
    else:
        ntraces = data.shape[-1]
        nsamps = data.shape[0]
        if np.size(interval) == 1:
            tr = [(np.arange(nsamps) * interval) for i in range(ntraces)]
            basis = np.transpose(np.asarray(tr))
        else:
            basis = interval
        tr = [np.arange(nsamps) + 1 for i in range(ntraces)]
        v_avg = np.cumsum(vmodel, axis=0) / np.transpose(tr)

    new_basis = basis / v_avg
    new_basis *= scale

    if np.size(interval_new) == 1:
        new_basis_lin = np.arange(np.amin(new_basis), np.amax(new_basis), interval_new)
    else:
        new_basis_lin = interval_new

    if np.ndim(data) == 1:
        inter = interp1d(new_basis, data,
                         bounds_error=False,
                         fill_value=data[-1],
                         kind=mode)
        result = inter(new_basis_lin)
    else:
        result = np.zeros((new_basis_lin.size, ntraces))
        for i in range(ntraces):
            inter = interp1d(new_basis[:, i], data[:, i],
                             bounds_error=False,
                             fill_value=data[-1, i],
                             kind=mode)
            result[:, i] += inter(new_basis_lin)

    if return_basis:
        field_names = ['data', 'basis']
        Conversion = namedtuple('Conversion', field_names)
        return Conversion(result, new_basis_lin)
    else:
        return result


def time_to_depth(data, vmodel, dt, dz, twt=True, mode="nearest", return_z=False):
    """
    Converts data from the time domain to the depth domain given a
    velocity model.

    Args:
        data (ndarray): The data to convert, will work with a 1 or 2D numpy
            numpy array. array(samples,traces).
        vmodel (ndarray): P-wave interval velocity model that corresponds to
            the data. Must be the same shape as data.
        dt (float): The sample interval of the input data [s], or an
            array of times.
        dz (float): The sample interval of the output data [m], or an
            array of depths.
        twt (bool): Use twt travel time, defaults to true.
        mode (str): What kind of interpolation to use, defaults to 'nearest'.
        return_z (bool): Whether to also return the new time basis.

    Returns
        ndarray: The data resampled in the depth domain.
    """
    if twt:
        scale = 0.5
    else:
        scale = 1.0

    # Do conversion with inverted velocity profile (slowness).
    return __convert(data,
                     vmodel=1. / vmodel,
                     interval=dt,
                     interval_new=dz,
                     scale=scale,
                     mode=mode,
                     return_basis=return_z,
                     )


def depth_to_time(data, vmodel, dz, dt, twt=True, mode="nearest", return_t=False):
    """
    Converts data from the depth domain to the time domain given a
    velocity model.

    Args:
        data (ndarray): The data to convert, will work with a 1 or 2D numpy
            numpy array. array(samples,traces).
        vmodel (ndarray): P-wave interval velocity model that corresponds to
            the data. Must be the same shape as data.
        dz (float): The sample interval of the input data [m].
        dt (float): The sample interval of the output data [s].
        twt (bool): Use twt travel time, defaults to true.
        mode (str): What kind of interpolation to use, defaults to 'nearest'.
        return_t (bool): Whether to also return the new time basis.

    Returns:
        The data resampled in the time domain.
    """
    if twt:
        scale = 2.0
    else:
        scale = 1.0

    return __convert(data,
                     vmodel=vmodel,
                     interval=dz,
                     interval_new=dt,
                     scale=scale,
                     mode=mode,
                     return_basis=return_t,
                     )
#=========================================================
# NOISE
#=========================================================
def noise_db(a, snr):
    """
    Takes an array of seismic amplitudes and SNR in dB.

     Args:
        a (array) : seismic amplitude array.
        snr (int): signal to noise ratio.

    Returns:  Noise array, the same shape as the input.

     Note: it does *not* return the input array with the noise added.

    """
    # Get the amplitude of the signal
    sigmean = rms(a)

    # Calculate the amp of the noise,
    # given the desired SNR
    noisemean = sigmean / 10.0**(snr/20.0)

    # Normal noise, centered on 0,
    # SD=sqrt(var), same shape as input
    noise = noisemean * np.random.normal(0.0, 1.0, a.shape)

    return noise


#================================================================================================
# SEISMIC ATTRIBUTES
#================================================================================================
"""
Mean-squared energy measurement.

:copyright: 2019 Agile Geoscience
:license: Apache 2.0
"""

def energy(traces, duration, dt=1):
    """
    Compute an mean-squared energy measurement on seismic data.

    The attribute is computed over the last dimension. That is, time should
    be in the last dimension, so a 100 inline, 100 crossline seismic volume
    with 250 time slices would have shape (100, 100, 250).

    Args:
        traces (ndarray): The data array to use for calculating energy.
        duration (float): the time duration of the window (in seconds), or
            samples if dt=1.
        dt (float): the sample interval of the data (in seconds). Defaults
            to 1 so duration can be in samples.
    Returns:
        ndarray: An array the same dimensions as the input array.
    """
    data = traces.astype(np.float).reshape(-1, traces.shape[-1])
    n_samples = int(duration / dt)
    window = np.ones(n_samples) / n_samples
    energy = convolve(data**2, window)
    return energy.reshape(traces.shape)

"""
A dip attribute, probably most useful for guiding other attributes.

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
def dipsteer(data,
             window_length,
             stepout,
             maxlag,
             overlap=1,
             dt=1,
             return_correlation=False):
    """
    Calculates a dip field by finding the maximum correlation between
    adjacent traces.

    :param data (ndarray): A 2D seismic section (samples,traces) used to
        calculate dip.
    :param window_length (float): The length [in ms] of the window to use.
    :param stepout (int): The number of traces on either side of each point
        to average when calculating the dip.
    :param maxlag (float): The maximum amount time lag to use when correlating
        the traces.
    :keyword overlap (float): The fractional overlap for each window. A value
        of 0 uses no redudant data, a value of 1 slides the dip correlator one
        sample at a time. Defaults to 1.
    :keyword dt (float): The time sample interval in ms.
    :keyword return_correlation (bool): Whether to return the correlation
        coefficients. If you choose True, you'll get a tuple, not an ndarray.
    :returns: a dip field [samples/trace] of the same shape as the input data
        (and optionally correlation coefficients, in which case you'll get a
        tuple of ndarrays back).
    """
    maxlag = int(maxlag)
    dip = np.zeros(data.shape)
    crcf = np.zeros(data.shape)

    window_length = int(np.floor(window_length / dt))

    # Force the window length to be odd for index tracking.
    if not (window_length % 2):
        window_length += 1

    # Define time windows.
    if overlap == 1:
        stride = 1
    else:
        stride = int(window_length * (1 - overlap))
    n_windows = np.ceil((data.shape[0] - window_length) / stride) + 1

    # Normalize each trace to the same RMS energy.
    norm_factor = np.sqrt(np.abs(energy(data, window_length)))
    norm_data = data / (norm_factor + 1e-9)  # To avoid div0 error.

    # Replace the 0/0 with 0.
    norm_data = np.nan_to_num(norm_data)

    # Mid point in the data which corresponds to zero dip.
    zero_dip = (np.floor(window_length / 2.0) + maxlag)

    s = stepout + 1

    # Loop over each trace we can do a full calculation for.
    for i in np.arange(s, data.shape[-1] - s):

        i = int(i)

        # Loop over each time window.
        for j in np.arange(0, n_windows):

            start = int((j * stride) + (maxlag))
            end = start + window_length

            # Don't compute last samples if we don't have a full window.
            if (end > (norm_data.shape[0]-maxlag)):
                break

            kernel = norm_data[start: end, i]

            dips_j, crcf_j = 0, 0

            # Correlate with adjacent traces.
            for k in np.arange(1, s):

                k = int(k)

                # Do the trace on the right.
                r_trace = norm_data[start - (k*maxlag): end + (k*maxlag), i+k]

                cor_r = np.correlate(kernel, r_trace, mode='same')

                if (np.amax(cor_r) < .1):
                    dip_r = 0
                else:
                    dip_r = (np.argmax(cor_r) - zero_dip) / k

                # Do the left trace.
                l_trace = norm_data[start - (k*maxlag): end + (k*maxlag), i-k]

                cor_l = np.correlate(kernel, l_trace, mode='same')

                if (np.amax(cor_l) < .1):
                    dip_l = 0
                else:
                    dip_l = -(np.argmax(cor_l) - zero_dip) / k

                dips_j += dip_r + dip_l
                crcf_j += np.argmin(cor_l) + np.argmin(cor_r)

            # Average the result
            dips_j /= (2. * stepout)
            crcf_j /= (2. * stepout)

            # Update the output
            dip[start: start+stride, i] = dips_j
            crcf[start: start+stride, i] = crcf_j

    if return_correlation:
        DipSteer = namedtuple('DipSteer', ['dip', 'correlation_coeff'])
        return DipSteer(dip, crcf)
    else:
        return dip
    
"""
A variance method to compute similarity. 

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
def similarity(traces, duration, dt=1, step_out=1, lag=0):
    """
    Compute similarity for each point of a seismic section using
    a variance method between traces.

    For each point, a kernel of n_samples length is extracted from a
    trace. The similarity is calculated as a normalized variance
    between two adjacent trace sections, where a value of 1 is
    obtained by identical if the traces are identical. The step out
    will decide how many adjacent traces will be used for each kernel,
    and should be increased for poor quality data. The lag determines
    how much neighbouring traces can be shifted when calculating
    similiarity, which should be increased for dipping data.

    :param traces: A 2D numpy array arranged as [time, trace].
    :param duration: The length in seconds of the window trace kernel
        used to calculate the similarity.
    :keyword step_out (default=1 ):
        The number of adjacent traces to
        the kernel to check similarity. The maximum
        similarity value will be chosen.
    :keyword lag (default=0):
        The maximum number of time samples adjacent traces
        can be shifted by. The maximum similarity of
        will be used.
    :keyword dt (default=1): The sample interval of the traces in sec.
        (eg. 0.001, 0.002, ...). Will default to one, allowing
        duration to be given in samples.
    """

    half_n_samples = int(duration / (2*dt))

    similarity_cube = np.zeros_like(traces, dtype='float32')
    traces = np.nan_to_num(traces)

    for j in np.arange(-lag, lag+1):

        for i in (np.arange(step_out)):
            for idx in range(similarity_cube.shape[0]):

                # Get the signal
                start_sig_idx = max(0, (idx+(j*(i+1))-half_n_samples))
                stop_sig_idx = min(similarity_cube.shape[0]-1, (idx-((i+1)*j))+half_n_samples)

                # Get the data
                start_data_idx = max(0, (idx - half_n_samples))
                end_data_idx = start_data_idx + (stop_sig_idx - start_sig_idx)

                if(end_data_idx > traces.shape[0]):
                    break

                signal = traces[start_sig_idx:stop_sig_idx, :]
                data = traces[start_data_idx:end_data_idx, :]

                squares = (signal*signal).sum(axis=0)

                squares_of_diff = ((signal[:,1+i:] - data[:, :-(1+i)])**2.).sum(axis=0)

                squares[squares == 0.0] = 0.001
                squares_of_diff[squares_of_diff == 0.0] = 0.001
                sim = 1.0 - np.sqrt(squares_of_diff) / ((np.sqrt(squares[1+i:]) + np.sqrt(squares[:-(1+i)]) ))

                similarity_cube[idx, (i+1):] = np.maximum(sim,
                                                         similarity_cube[idx, (i+1):])

    return similarity_cube
    
"""
Spectrogram.

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
def spectrogram(data, window_length,
                dt=1.0,
                window_type='boxcar',
                overlap=0.5,
                normalize=False,
                zero_padding=0):
    """
    Calculates a spectrogram using windowed STFTs.

    :param data: 1D numpy array to process into spectra.
    :param window_length: The length of the window in seconds if
                          dt is set, otherwise in samples. Will
                          zero pad to the closest power of two.
    :keyword window_type: A string specifying the type of window to
                          use for the STFT. The same input as
                          scipy.signal.get_window
    :keyword dt: The time sample interval of the trace. Defaults to
                 1, which allows window_length to be in seconds.

    :keyword overlap: The fractional overlap for each window.
                      A value of 0 uses no redudant data, a value of 1
                      slides the STFT window one sample at a time.
                      Defaults to 0.5
    :keyword normalize: Normalizes the each spectral slice to have
                        unity energy.
    :keyword zero_padding: The amount of zero padding to when creating
                           the spectra.

    :returns: A spectrogram of the data ([time, freq]).
            ( 2D array for 1D input )
    
     See Also
    --------
    spectraldecomp : Spectral decomposition
    
    """
    # Make the base window
    window_n = int(np.floor(window_length / dt))
    pad = int(np.floor(zero_padding / dt))
    window = get_window(window_type, window_n)

    # Calculate how many STFTs we need to do.
    stride = int(window.size * (1 - overlap) + 1)
    n_windows = int(np.ceil((data.size - window.size) / stride) + 1)

    # Pad window to power of 2
    padded_window = np.zeros(next_pow2(window.size+pad))

    # Init the output array
    output = np.zeros([n_windows, int(padded_window.size // 2)])

    # Do the loop
    for i in range(int(n_windows)):

        start = int(i * stride)
        end = start + int(window.size)

        # Do not compute last samples if we don't have a full window
        if (end > data.size-1):
            break

        padded_window[0:window.size] = window*data[start:end]
        spect = (2. * np.absolute(fft(padded_window)) /
                 window.size)[0:int(padded_window.size // 2)]

        if normalize:
            output[i, :] = spect / np.sum(spect**2)

        else:
            output[i, :] = spect

    return output

"""
Spectral decomposition

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
def spectraldecomp(data,
                   f=(0.1, 0.25, 0.4),
                   window_length=32,
                   dt=1,
                   window_type='hann',
                   overlap=1,
                   normalize=False):
    """
    Uses the STFT to decompose traces into normalized spectra. Only
    frequencies defined by the user will be output. Using 3
    frequencies will work for RGB color plots.

    :param data: A 1/2D array (samples, traces) of data that will
                 be decomposed.
    :keyword f: A list of frequencies to select from the spectra.
    :keyword window_length: The length of fft window to use for
                            the STFTs. Defaults to 32. Can be
                            provided in seconds if dt is specified.
    :keyword dt: The time sample interval of the traces.
    :keyword window_type: The type of window to use for the STFT. The
                          same input as scipy.signal.get_window.
    :keyword overlap: The fraction of overlap between adjacent
                      STFT windows

    :keyword normalize: Normalize the energy in each STFT window

    :returns: an array of shape (samples, traces, f)
    """

    # Do the 1D case
    if len(data.shape) == 1:
        ntraces = 1
    else:
        ntraces = data.shape[-1]

    if overlap > 1:
        overlap = 1

    zp = 4 * window_length

    # TODO We should think about removing these for loops
    for i in range(ntraces):

        if(ntraces == 1):
            spect = spectrogram(data, window_length, dt=dt,
                                window_type=window_type, overlap=overlap,
                                normalize=normalize, zero_padding=zp)
            if(i == 0):
                output = np.zeros((spect.shape[0], len(f)))
        else:
            spect = spectrogram(data[:, i], window_length, dt=dt,
                                window_type=window_type, overlap=overlap,
                                normalize=normalize, zero_padding=zp)
            if(i == 0):
                output = np.zeros((spect.shape[0], ntraces, len(f)))

        res = ((1. / dt) / 2.) / spect.shape[-1]

        # TODO again, we should think about removing this loop
        for j in range(len(f)):

            index = int(f[j] / res)

            if(ntraces == 1):
                output[:, j] = spect[:, index]
            else:
                output[:, i, j] = spect[:, index]

    return(output)
