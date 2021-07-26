#-----------------------------------------------------------------------
# inspired by compact plotting style written by Alessandro Amato del Monte, 2016
# with some modifications

# Most of the functions in this file were originally written by therefore 
# are credited to Alessandro Amato del Monte (aadm), 2016 and 2017
#
# aadm's github repo and projects are excellent sources for seismic QI workflow and 
# can be obtained from here https://github.com/aadm or 
#
# most of aadm's options in the original codes were removed and simplified for my class 
# students will discuss their uses in the class (-adi w, 2021)

#-----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
from matplotlib import cm
from brugeslibrary import shuey, akirichards
#---------------------- simple aki richard style plot-------------------

def basicgassmann(vp1, vs1, rho1, rho_fl1, k_fl1, rho_fl2, k_fl2, k0, phi):
    vp1 = vp1; vs1 = vs1
    rho2 = rho1-phi*rho_fl1+phi*rho_fl2
    mu1 = rho1*vs1**2.
    k1 = rho1*vp1**2-(4./3.)*mu1
    kdry= (k1 * ((phi*k0)/k_fl1+1-phi)-k0) / ((phi*k0)/k_fl1+(k1/k0)-1-phi)
    k2 = kdry + (1- (kdry/k0))**2 / ( (phi/k_fl2) + ((1-phi)/k0) - (kdry/k0**2) )
    mu2 = mu1
    vp2 = np.sqrt(((k2+(4./3)*mu2))/rho2)
    vs2 = np.sqrt((mu2/rho2))
    return [vp2, vs2, rho2, k2, kdry]

def AR3term(vp1, vs1, rho1, vp2, vs2, rho2, theta):
    a = np.radians(theta)
    p = np.sin(a)/vp1
    dvp = vp2-vp1
    dvs = vs2-vs1
    drho = rho2-rho1
    vp  = np.mean([vp1,vp2])
    vs  = np.mean([vs1,vs2])
    rho = np.mean([rho1,rho2])
    A = 0.5*(1-4*p**2*vs**2)*drho/rho
    B = 1/(2*np.cos(a)**2) * dvp/vp
    C = 4*p**2*vs**2*dvs/vs
    R = A + B - C
    return R, A, B, C

def avoar3(vp1,vs1,rho1,vp2,vs2,rho2,angmin=0,angmax=30, waveletphase=0):
    n_samples = 200
    gain=40
    interface=int(n_samples/2)
    ang = np.arange(angmin,angmax+1,1)
    z = np.arange(n_samples)
    ip, vpvs = (np.zeros(n_samples) for _ in range(2))
    ip[:interface] = vp1*rho1
    ip[interface:] = vp2*rho2
    vpvs[:interface] = vp1/vs1
    vpvs[interface:] = vp2/vs2
    avoar3 = AR3term(vp1,vs1,rho1,vp2,vs2,rho2,ang)
    _,A,B,_ = AR3term(vp1,vs1,rho1,vp2,vs2,rho2,40)
    _,wavelet = ricker(f=10, length=.200, dt=0.001, waveletphase=0)
    rc, syn = (np.zeros((n_samples,ang.size)) for _ in range(2))
    rc[interface,:]=  avoar3
    for i in range(ang.size):
        syn[:,i] = np.convolve(rc[:,i],wavelet,mode='same')

def avoplot1(vp1,vs1,rho1,vp2,vs2,rho2,thetamin,thetamax, opt="aki"):
    f, l, dt, ph = 20, 200, 0.5, 0
    t = np.linspace(-l/2, (l-dt)/2, int(l/dt))/1000
    wavelet = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    n_samples, gain = 400, 20
    half=n_samples/2
    interface=int(half)
    ang = np.arange(thetamin,thetamax+1,1)
    z = np.arange(n_samples)
    ip, vpvs = (np.zeros(n_samples) for _ in range(2))
    ip[:interface] = vp1*rho1
    ip[interface:] = vp2*rho2
    vpvs[:interface] = vp1/vs1
    vpvs[interface:] = vp2/vs2
    if opt=="shuey":
        avo = shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1=ang, terms=False, return_gradient=False)
        I,G=shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False, return_gradient=True)
    elif opt=="aki":
        avo = akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta=ang, terms=False)
        A,B,C,D = akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta=ang, terms=True)
    rc, syn = (np.zeros((n_samples,ang.size)) for _ in range(2))
    rc[interface,:]= avo
    for i in range(ang.size):
        syn[:,i] = np.convolve(rc[:,i],wavelet,mode='same')
    # best plot with %config InlineBackend.figure_format = 'png' setting
    f=plt.subplots(figsize=(16, 6), facecolor="white")
    mpl.style.use('seaborn')
    ax0 = plt.subplot2grid((1,5), (0,0), colspan=1)
    ax1 = plt.subplot2grid((1,5), (0,1), colspan=1)
    ax2 = plt.subplot2grid((1,5), (0,2), colspan=1)
    ax3 = plt.subplot2grid((1,5), (0,3), colspan=2)
    ax0.plot(ip, z, 'darkblue', lw=6)
    ax0.set_xlabel('AI [m/s*g/cc]', size=16)
    ax0.margins(x=0.2)
    ax1.plot(vpvs, z, 'darkblue', lw=6)
    ax1.set_xlabel('Vp/Vs', size=16)
    ax1.margins(x=0.2)
    ax1.axhline(half, color='k', lw=0.4)
    for i in range(0,int(ang.size),5):
        trace=gain*syn[:,i] / np.max(np.abs(syn))
        ax2.plot(i+trace,z,'k',lw=0.6)
        ax2.fill_betweenx(z,trace+i,i,where=trace+i>i,facecolor='darkblue', lw=0, alpha=0.6)
        ax2.fill_betweenx(z,trace+i,i,where=trace+i<i,facecolor='crimson', lw=0, alpha=0.6)
        ax2.set_xticklabels([])
    ax2.margins(x=0.05)
    ax2.set_xlabel('Synthetics', size=16)
    ax3.axhline(half, color='k', lw=0.4)
    ax3.plot(ang, avo,'darkblue', lw=6)
    ax3.axhline(0, color='k', lw=1)
    ax3.set_xlabel('Angle('r'$\theta)$', size=16)
    ax3.margins(y=0.02),  ax3.set_ylim(-0.5,0.5)
    ax3.yaxis.tick_right(), ax3.xaxis.tick_top()
    ax3.tick_params(axis='both', colors='k')
    for aa in [ax0, ax1, ax2]:
        aa.invert_yaxis()
        aa.xaxis.tick_top()
        plt.setp(aa.xaxis.get_majorticklabels(), color='k', rotation=0, fontsize=12)
        aa.set_yticklabels([])
        aa.axhline(half, color='k', lw=0.4)
    amin="min=%i" % thetamin
    amax="max=%i" % thetamax
    ax2.annotate(amin, xy=(0,0), xytext=(-30, 35), textcoords='offset points', color='k', size=12, va='top')
    ax2.annotate(amax, xy=(45,0), xytext=(-25, 35), textcoords='offset points', color='k', size=12, va='top')
    plt.text(-88, 0.42, 'Vp1={:.2f}'.format(vp1),fontsize=12)
    plt.text(-88, 0.36, 'Vs1={:.2f}'.format(vs1),fontsize=12)
    plt.text(-88, 0.30, 'Rho1={:3.2f}'.format(rho1),fontsize=12)
    plt.text(-84, -0.26, 'Vp2={:.2f}'.format(vp2),fontsize=12)
    plt.text(-84, -0.32, 'Vs2={:.2f}'.format(vs2),fontsize=12)
    plt.text(-84, -0.38, 'Rho2={:3.2f}'.format(rho2),fontsize=12)
    
    
#----------------- this one with intercept gradient shuey---------------

def avoplot2(vp1,vs1,rho1,vp2,vs2,rho2,thetamin,thetamax, opt="shuey"):
    f, l, dt, ph = 20, 200, 0.5, 0
    t = np.linspace(-l/2, (l-dt)/2, int(l/dt))/1000
    wavelet = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    n_samples, gain = 400, 20
    half=n_samples/2
    interface=int(half)
    ang = np.arange(thetamin,thetamax+1,1)
    z = np.arange(n_samples)
    ip, vpvs = (np.zeros(n_samples) for _ in range(2))
    ip[:interface] = vp1*rho1
    ip[interface:] = vp2*rho2
    vpvs[:interface] = vp1/vs1
    vpvs[interface:] = vp2/vs2
    if opt=="shuey":
        avo = shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1=ang, terms=False, return_gradient=False)
        I,G=shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False, return_gradient=True)
    elif opt=="aki":
        avo = akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta1=ang, terms=False)
        A,B,C,D = akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta1=ang, terms=True)
    rc, syn = (np.zeros((n_samples,ang.size)) for _ in range(2))
    rc[interface,:]= avo
    for i in range(ang.size):
        syn[:,i] = np.convolve(rc[:,i],wavelet,mode='same')
    # best plot with %config InlineBackend.figure_format = 'png' setting
    f=plt.subplots(figsize=(16, 5), facecolor="white")
    mpl.style.use('seaborn')
    ax0 = plt.subplot2grid((1,7), (0,0), colspan=1)
    ax1 = plt.subplot2grid((1,7), (0,1), colspan=1)
    ax2 = plt.subplot2grid((1,7), (0,2), colspan=1)
    ax3 = plt.subplot2grid((1,7), (0,3), colspan=2)
    ax4 = plt.subplot2grid((1,7), (0,5), colspan=2)
    ax0.plot(ip, z, 'darkblue', lw=6)
    ax0.set_xlabel('AI [m/s*g/cc]', size=14)
    ax0.margins(x=0.2)
    ax1.plot(vpvs, z, 'darkblue', lw=6)
    ax1.set_xlabel('Vp/Vs', size=14)
    ax1.margins(x=0.2)
    ax1.axhline(half, color='k', lw=0.4)
    for i in range(0,int(ang.size),5):
        trace=gain*syn[:,i] / np.max(np.abs(syn))
        ax2.plot(i+trace,z,'k',lw=0.6)
        ax2.fill_betweenx(z,trace+i,i,where=trace+i>i,facecolor='darkblue', lw=0, alpha=0.6)
        ax2.fill_betweenx(z,trace+i,i,where=trace+i<i,facecolor='crimson', lw=0, alpha=0.6)
        ax2.set_xticklabels([])
    ax2.margins(x=0.05)
    ax2.set_xlabel('Amplitudes', size=14)
    ax3.axhline(half, color='k', lw=0.4)
    ax3.plot(ang, avo,'darkblue', lw=6)
    ax3.axhline(0, color='k', lw=1)
    ax3.set_xlabel('Angle('r'$\theta)$', size=14)
    ax3.margins(y=0.05)
    ax3.set_ylim(-0.5,0.5)
    ax3.yaxis.tick_right()
    ax3.xaxis.tick_top()
    ax3.yaxis.tick_right()
    ax3.xaxis.tick_top()
    ax3.set_yticklabels([])
    ax3.axhline(half, color='k', lw=0.4)
    ax4.plot(I,G,'o',ms=20,mfc='darkblue',mew=1)
    ax4.axhline(0, color='k', lw=1), ax4.axvline(0, color='k', lw=1)
    ax4.set_xlabel('intercept', fontsize=14), ax4.set_ylabel('gradient', fontsize=14)
    scaleig=0.5
    ax4.set_xlim(-scaleig,scaleig), ax4.set_ylim(-scaleig,scaleig)
    ax4.xaxis.set_label_position('bottom'), ax4.xaxis.tick_bottom(), ax4.tick_params(axis="x", labelsize=12)
    ax4.yaxis.set_label_position('right'), ax4.yaxis.tick_right(), ax4.tick_params(axis="y", labelsize=12)
    for aa in [ax0, ax1, ax2]:
        aa.invert_yaxis()
        aa.xaxis.tick_top()
        plt.setp(aa.xaxis.get_majorticklabels(), rotation=0, fontsize=12)
        aa.set_yticklabels([])
        aa.axhline(half, color='k', lw=0.4)
    amin="%i" % thetamin
    amax="%i" % thetamax
    ax2.annotate(amin, xy=(15,0), xytext=(-26, 30), textcoords='offset points', size=12, va='top')
    ax2.annotate(amax, xy=(50,0), xytext=(-10, 30), textcoords='offset points', size=12, va='top')
    plt.text(-3.6, 0.42, 'Vp1={:.2f}'.format(vp1),fontsize=12)
    plt.text(-3.6, 0.36, 'Vs1={:.2f}'.format(vs1),fontsize=12)
    plt.text(-3.6, 0.30, 'Rho1={:3.2f}'.format(rho1),fontsize=12)
    plt.text(-3.6, -0.24, 'Vp2={:.2f}'.format(vp2),fontsize=12)
    plt.text(-3.6, -0.30, 'Vs2={:.2f}'.format(vs2),fontsize=12)
    plt.text(-3.6, -0.36, 'Rho2={:3.2f}'.format(rho2),fontsize=12)
    plt.text(-3.6, 0.22, 'AI1={:.0f}'.format(vp1*rho1),fontsize=12)
    plt.text(-3.6, 0.16, 'VpVs1={:3.2f}'.format(vp1/vs1),fontsize=12)
    plt.text(-3.6, -0.42, 'AI2={:.0f}'.format(vp2*rho2),fontsize=12)
    plt.text(-3.6, -0.48, 'VpVs2={:3.2f}'.format(vp2/vs2),fontsize=12)



#------------------------ near-mid-far traces plot----------------------

def avoplot3(vp1,vs1,rho1,vp2,vs2,rho2,thetamin,thetamax, opt="shuey"):
    f, l, dt, ph = 20, 200, 0.5, 0
    t = np.linspace(-l/2, (l-dt)/2, int(l/dt))/1000
    wavelet = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    n_samples, gain = 400, 20
    half=n_samples/2
    interface=int(half)
    nearmax=((thetamax-thetamin)/3)
    farmin=(2*(thetamax-thetamin)/3)
    ang = np.arange(thetamin,thetamax+1,1)
    near = np.arange(thetamin,nearmax+1,1)
    mid = np.arange(nearmax,farmin+1,1)
    far = np.arange(farmin,thetamax+1,1)
    z = np.arange(n_samples)
    vp, vs, rho = (np.zeros(n_samples) for _ in range(3))
    vp[:interface] = vp1
    vp[interface:] = vp2
    vs[:interface] = vs1
    vs[interface:] = vs2
    rho[:interface] = rho1
    rho[interface:] = rho2
    ip, vpvs = (np.zeros(n_samples) for _ in range(2))
    ip[:interface] = vp1*rho1
    ip[interface:] = vp2*rho2
    vpvs[:interface] = vp1/vs1
    vpvs[interface:] = vp2/vs2
    if opt=="shuey":
        avo = shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1=ang, terms=False, return_gradient=False)
        navo = shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1=near, terms=False, return_gradient=False)
        mavo = shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1=mid, terms=False, return_gradient=False)
        favo = shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1=far, terms=False, return_gradient=False)
        I,G=shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False, return_gradient=True)
    elif opt=="aki":
        avo = akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta1=ang, terms=False)
        navo = akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta1=near, terms=False)
        mavo = akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta1=mid, terms=False)
        favo = akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta1=far, terms=False)
        A,B,C,D = akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta1=ang, terms=True)
    rc, syn = (np.zeros((n_samples,ang.size)) for _ in range(2))
    rcn, nearsyn = (np.zeros((n_samples,near.size)) for _ in range(2))
    rcm, midsyn = (np.zeros((n_samples,mid.size)) for _ in range(2))
    rcf, farsyn = (np.zeros((n_samples,far.size)) for _ in range(2))
    rc[interface,:]= avo
    rcn[interface,:]= navo
    rcm[interface,:]= mavo
    rcf[interface,:]= favo
    for i in range(ang.size):
        syn[:,i] = np.convolve(rc[:,i],wavelet,mode='same')
    for i in range(near.size):
        nearsyn[:,i] = np.convolve(rcn[:,i],wavelet,mode='same')
    for i in range(mid.size):
        midsyn[:,i] = np.convolve(rcm[:,i],wavelet,mode='same')
    for i in range(far.size):
        farsyn[:,i] = np.convolve(rcf[:,i],wavelet,mode='same')
    f1 = plt.figure(1,figsize = (12,4),facecolor="white")
    mpl.style.use('seaborn')
    ax1 = f1.add_axes([0.1, 0.1, 0.1, 0.8])
    ax1.plot( vp, z,'darkblue', lw=4)
    ax1.set_xlabel('Vp.km/s', size=14)
    ax1.set_xlim(3.8,4.8)
    ax1.set_ylabel('unit')
    ax2 = f1.add_axes([0.2 , 0.1, 0.1, 0.8])
    ax2.plot( vs, z,'darkblue', lw=4)
    ax2.set_xlabel('Vs.km/s', size=14)
    ax2.set_xlim(2.5,3.5)
    ax2.set_yticklabels('')
    ax3 = f1.add_axes([0.3 , 0.1, 0.1, 0.8])
    ax3.plot( rho, z,'darkblue', lw=4)
    ax3.set_xlabel('Rho.g/cc', size=14)
    ax3.set_xlim(2.65,2.75)
    ax3.set_yticklabels('')
    ax4 = f1.add_axes([0.41, 0.1, 0.1, 0.8])
    ax4.plot( ip, z,'darkblue', lw=4)
    ax4.set_xlabel('AI [m/s*g/cc]', size=14)
    ax4.set_xlim(10,13)
    ax4.set_yticklabels('')
    ax5 = f1.add_axes([0.5, 0.1, 0.1, 0.8])
    ax5.plot( vpvs, z,'darkblue', lw=4)
    ax5.set_xlabel('Vp/Vs', size=14)
    ax5.set_xlim(1.4,1.44)
    ax5.set_yticklabels('')
    ax6 = f1.add_axes([0.61, 0.1, 0.1, 0.8])
    for i in range(0,int(near.size),4):
        trace=gain*nearsyn[:,i] / np.max(np.abs(syn))
        ax6.plot(i+trace,z,'k',lw=0.6)
        ax6.fill_betweenx(z,trace+i,i,where=trace+i>i,facecolor='darkblue', lw=0, alpha=0.6)
        ax6.fill_betweenx(z,trace+i,i,where=trace+i<i,facecolor='crimson', lw=0, alpha=0.6)
    ax6.set_xlabel('Near Traces', size=14)
    ax7 = f1.add_axes([0.72, 0.1, 0.1, 0.8])
    for i in range(0,int(mid.size),3):
        trace=gain*midsyn[:,i] / np.max(np.abs(syn))
        ax7.plot(i+trace,z,'k',lw=0.6)
        ax7.fill_betweenx(z,trace+i,i,where=trace+i>i,facecolor='darkblue', lw=0, alpha=0.6)
        ax7.fill_betweenx(z,trace+i,i,where=trace+i<i,facecolor='crimson', lw=0, alpha=0.6)
    ax7.set_xlabel('Mid Traces', size=14)
    ax8 = f1.add_axes([0.83, 0.1, 0.1, 0.8])
    for i in range(0,int(far.size),2):
        trace=gain*farsyn[:,i] / np.max(np.abs(syn))
        ax8.plot(i+trace,z,'k',lw=0.6)
        ax8.fill_betweenx(z,trace+i,i,where=trace+i>i,facecolor='darkblue', lw=0, alpha=0.6)
        ax8.fill_betweenx(z,trace+i,i,where=trace+i<i,facecolor='crimson', lw=0, alpha=0.6)
    ax8.set_xlabel('Far Traces', size=14)
    ax9 = f1.add_axes([0.94, 0.1, 0.2, 0.8])
    for i in range(0,int(ang.size),4):
        trace=gain*syn[:,i] / np.max(np.abs(syn))
        ax9.plot(i+trace,z,'k',lw=0.6)
        ax9.fill_betweenx(z,trace+i,i,where=trace+i>i,facecolor='darkblue', lw=0, alpha=0.6)
        ax9.fill_betweenx(z,trace+i,i,where=trace+i<i,facecolor='crimson', lw=0, alpha=0.6)
    ax9.set_xlabel('Full Traces', size=14)
    for aa in [ax1, ax2, ax3, ax4, ax5]:
        aa.invert_yaxis()
        aa.xaxis.tick_top()
        aa.yaxis.tick_left()
        aa.xaxis.set_label_position('top')
        aa.minorticks_on()
        plt.setp(aa.xaxis.get_majorticklabels(), rotation=0, fontsize=10)
        aa.tick_params(axis='x', size=4)
    for bb in [ax6, ax7, ax8, ax9]:
        bb.set_xticklabels([])
        bb.margins(x=0.05)
        bb.set_yticklabels('')
        bb.xaxis.set_label_position('top')
        bb.spines['top'].set_position(('outward',20))      
    amin="%i" % thetamin
    anmax="%i" % nearmax
    fmin="%i" % farmin
    fmax="%i" % thetamax
    ax6.annotate(amin, xy=(10,0), xytext=(-30, 242), textcoords='offset points', size=12, va='top')
    ax6.annotate(anmax, xy=(15,0), xytext=(0, 242), textcoords='offset points', size=12, va='top')
    ax7.annotate(anmax, xy=(5,0), xytext=(-30, 242), textcoords='offset points', size=12, va='top')
    ax7.annotate(fmin, xy=(15,0), xytext=(0, 242), textcoords='offset points', size=12, va='top')
    ax8.annotate(fmin, xy=(5,0), xytext=(-30, 242), textcoords='offset points', size=12, va='top')
    ax8.annotate(fmax, xy=(15,0), xytext=(0, 242), textcoords='offset points', size=12, va='top')
    ax9.annotate(amin, xy=(10,0), xytext=(-30, 242), textcoords='offset points', size=12, va='top')
    ax9.annotate(fmax, xy=(45,0), xytext=(0, 242), textcoords='offset points', size=12, va='top')

