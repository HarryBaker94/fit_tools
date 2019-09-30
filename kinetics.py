#This module contains useful functions for fitting data pertaining to kinetic transients
"""
Created 13/09/2018
Fitting tools for kinetic transients

Just some functions for doing reconvolution and tail fits
Also allows adds functions for uneven time grids
"""

import numpy as np
import lmfit
from decorator import decorator
from scipy.interpolate import interp1d


def convolve(arr, kernel):
    """
    Convolution of array with kernel.
    """
    #logger.debug("Convolving...")
    npts = min(len(arr), len(kernel))
    pad  = np.ones(npts)
    tmp  = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
    norm = np.sum(kernel)
    out  = np.convolve(tmp, kernel, mode='valid')
    noff = int((len(out) - npts)/2)
    return out[noff:noff+npts]/norm


def gauss_kernel(x, t0, irf):
    """
    Gaussian convolution kernel.

    Parameters
    ----------
    x : array-like
        Independant variable
    t0 : array-like
        t0 offset
    irf : array-like
        Irf gaussian width (sigma)
    """
    midp = 0.5*(np.max(x)+np.min(x))
    return lmfit.lineshapes.gaussian(x, 1, midp+t0, irf) # you can replace with your favorite gaussian implementation. Make sure it's normalized.

def regrid(idx):
    """
    Decorator factory to compute a model on a constant grid, then interpolate.

    This is to be used for reconvolution fits when the independant axis isn't
    evently spaced. This function returns a decorator. You should call the
    result of this function with the model to regrid. The constant grid

    Parameters
    ----------
    idx : int
        Index of variable to regrid in client function.

    Returns
    -------
    regridder : decorator

    Example
    -------
    ```
    def model(x, amp, tau, t0, sig):
        # Convolution assumes constant grid spacing.
        return convolve(step(x)*exp_decay(x, amp, tau), gauss_kernel(x, t0, sig))

    deco = regrid(1)
    regridded = deco(model)
    # Or, on a single line
    regridded = regrid(1)(model) # compute on first axis
    # Or, during definition
    @regrid(1)
    def model(x, *args):
        ...
    ```
    """
    #logger.debug("Applying 'regrid' decorator")
    def _regrid(func, *args, **kw):
        #logger.debug("Regridding func {}".format(func.__name__))
        x = args[idx]
        #print("regridding...")
        mn, mx = np.min(x), np.max(x)
        extension=1
        margin = (mx-mn)*extension
        dx = np.abs(np.min(x[1:]-x[:-1]))
        #print("regrid args", args)
        #print("regrid kw", kw)
        #print("regrid func", func)
        grid = np.arange(mn-margin, mx+margin+dx, dx)
        args = list(args)
        args[idx] = grid
        y = func(*args, **kw)
        #print("y", y)
        intrp = interp1d(grid, y, kind=3, copy=False, assume_sorted=True)
        return intrp(x)
    return decorator(_regrid)

def step(x):
    """Heaviside step function."""
    step = np.ones_like(x, dtype='float')
    step[x<0] = 0
    step[x==0] = 0.5
    return step

def exp_decay(t, a, tau):
    return step(t)*a*np.exp(-t/tau)

def exp_conv(t,a,tau,t0,irf):
    return convolve(exp_decay(t,a,tau),gauss_kernel(t,t0,irf))

def biexp_decay(t, a0, a1, tau0, tau1):
    return exp_decay(t, a0, tau0)+exp_decay(t, a1, tau1)

def biexp_conv(t, a0, a1, tau0, tau1, t0, irf):
    return convolve(
        biexp_decay(t, a0, a1, tau0, tau1),
        gauss_kernel(t, t0, irf))

def triexp_decay(t,a0,a1,a2,tau0,tau1,tau2):
    return exp_decay(t, a0, tau0)+exp_decay(t, a1, tau1)+exp_decay(t,a2,tau2)

def triexp_conv(t,a0,a1,a2,tau0,tau1,tau2,t0,irf):
    return convolve(triexp_decay(t,a0,a1,a2,tau0,tau1,tau2),gauss_kernel(t,t0,irf))

def tau_av(a0,tau0,a1,tau1,a2,tau2):
    if a2 is False or tau2 is False:
        tau_av = (a0*tau0**2+a1*tau1**2)/(a0*tau0+a1*tau1)
        return tau_av
    tau_av = (a0*tau0**2+a1*tau1**2+a2*tau2**2)/(a0*tau0+a1*tau1+a2*tau2)
    return tau_av