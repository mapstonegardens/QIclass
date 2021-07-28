import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
from scipy.signal import hilbert
import scipy as sp
import scipy.ndimage as snd
from brugeslibrary import shuey, akirichards

def ricker(f, l, dt, wphase):
    t = np.linspace(-l/2, (l-dt)/2, int(l/dt))/1000
    zeroph = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    h = hilbert(zeroph)
    rotdeg = wphase
    theta = rotdeg*np.pi/180
    y = np.cos(theta)*h.real-np.sin(theta)*h.imag
    return t, y
def outputmodelvpvsrho(model, elprop):
    model_vp=np.zeros(model.shape)
    model_vs=np.zeros(model.shape)
    model_rho=np.zeros(model.shape)
    code = 1
    for i in elprop:
        model_vp[model==code]  = i[0]
        model_vs[model==code]  = i[1]
        model_rho[model==code] = i[2]
        code += 1
    return model_vp,model_vs,model_rho
def rcshuey(vp1, vs1, rho1, vp2, vs2, rho2, theta):
    a = np.radians(theta)
    dvp = vp2-vp1
    dvs = vs2-vs1
    drho = rho2-rho1
    vp  = np.mean([vp1,vp2])
    vs  = np.mean([vs1,vs2])
    rho = np.mean([rho1,rho2])
    I = 0.5*(dvp/vp + drho/rho)
    G = 0.5*(dvp/vp) - 2*(vs**2/vp**2)*(drho/rho+2*(dvs/vs))
    F = 0.5*(dvp/vp)
    R = I + G*np.sin(a)**2 + F*(np.tan(a)**2-np.sin(a)**2)
    return R
def partialstacks(model_vp,model_vs,model_rho,ang):
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
def synthfrimage(rc,wavelet):
    from scipy.ndimage.filters import convolve1d
    nt=np.size(wavelet)
    [n_samples, n_traces] = rc.shape
    synth=np.zeros((n_samples+nt-1, n_traces))
    synth=convolve1d(rc,wavelet,axis=0)
    return synth