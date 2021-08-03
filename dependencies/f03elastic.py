import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from brugeslibrary import backus, moving_average, moving_avg_fft
import matplotlib.colors as colors

#----plot well section, elastic logs---------------------------
def elastic_sect(datawell,top_plot,base_plot,marker,markerdepth):
    logs=datawell[(datawell.index >= top_plot) & (datawell.index <= base_plot)]
    f, ax = plt.subplots(nrows=1, ncols=5, figsize=(12,6), sharey=True, dpi=80)
    f.subplots_adjust(top=0.2,wspace=0.1)
    for aa in ax:
        aa.set_ylim(top_plot, base_plot)
        aa.invert_yaxis()
        aa.yaxis.grid(True)
        aa.xaxis.grid(True)
        aa.xaxis.set_ticks_position('bottom')
        aa.xaxis.set_label_position('top')
        aa.grid(True,linestyle=':')
        for (i,j) in zip(markerdepth,marker):
            if ((i>=top_plot) and (i<=base_plot)):
                aa.axhline(y=i, linewidth=0.4, color='b')
                ax[0].text(1, i ,j, color='b', size=14)
    #kolom: Vp
    ax01=ax[0]
    vpscmin=0.9*(logs.Vp.min())
    vpscmax=1.1*(logs.Vp.max())
    ax01.set_xlim(vpscmin,vpscmax)
    ax01.set_xlabel("Vp km/s", color='k', size=14)
    ax01.plot(logs.Vp, logs.index, label='Vp', color='k')
    #kolom: Vs
    ax11=ax[1]
    vsscmin=0.9*(logs.Vs.min())
    vsscmax=1.1*(logs.Vs.max())
    ax11.set_xlim(vsscmin,vsscmax)
    ax11.set_xlabel("Vs km/s", color='k', size=14)
    ax11.plot(logs.Vs, logs.index, label='Vs', color='k')
    #kolom: Rho
    ax21=ax[2]
    rhoscmin=0.9*(logs.Rho.min())
    rhoscmax=1.1*(logs.Rho.max())
    ax21.set_xlim(rhoscmin,rhoscmax)
    ax21.set_xlabel("Rho g/cc", color='k', size=14)
    ax21.plot(logs.Rho, logs.index, label='Rho', color='k')
    #kolom: Vsh
    ax31=ax[3]
    ax31.set_xlim(0,1)
    ax31.set_xlabel("Vsh v/v", color='k', size=14)
    ax31.plot(logs.Vsh, logs.index, label='Vsh', color='k')
    #kolom: Properties
    ax41=ax[4].twiny()
    ax41.set_xlim(0.2,1)
    ax41.spines['top'].set_position(('outward',5))
    ax41.set_xlabel("Sat", color='k', size=14)
    ax41.plot(logs.Swt, logs.index, label='Sat', color='darkgreen')
    ax41.tick_params(axis='x', colors='darkgreen')
    ax41.fill_betweenx(logs.index,logs.Swt,logs.Swt.max(), facecolor='lightgreen', alpha=0.2)
    ax42=ax[4].twiny()
    ax42.set_xlim(0,0.5)
    ax42.plot(logs.Phit, logs.index, label='Phit', color='k') 
    ax42.spines['top'].set_position(('outward',40))
    ax42.set_xlabel('Phit',color='k', size=14)    
    ax42.tick_params(axis='x', colors='k')
    for bb in [ax01, ax11, ax21, ax31]:
        bb.spines['top'].set_position(('outward',5))
        bb.tick_params(axis='x', colors='k')
    
    f.tight_layout() 

#--------------complete backus ----------------
def linerbackus(vp, vs, rho, lb, dz):
    lam = moduli.lam(vp, vs, rho)
    mu = moduli.mu(vp, vs, rho)
    a = rho * np.power(vp, 2.0)
    A1 = 4 * moving_average(mu*(lam+mu)/a, lb/dz, mode='same')
    A = A1 + np.power(moving_average(lam/a, lb/dz, mode='same'), 2.0)\
        / moving_average(1.0/a, lb/dz, mode='same')
    C = 1.0 / moving_average(1.0/a, lb/dz, mode='same')
    F = moving_average(lam/a, lb/dz, mode='same')\
        / moving_average(1.0/a, lb/dz, mode='same')
    L = 1.0 / moving_average(1.0/mu, lb/dz, mode='same')
    M = moving_average(mu, lb/dz, mode='same')
    R = moving_average(rho, lb/dz, mode='same')
    vp0 = np.sqrt(C / R)
    vs0 = np.sqrt(L / R)
    ptemp = np.pi * np.log(vp0 / vp) / (np.log(vp0 / vp) + np.log(lb/dz))
    Qp = 1.0 / np.tan(ptemp)
    stemp = np.pi * np.log(vs0 / vs) / (np.log(vs0 / vs) + np.log(lb/dz))
    Qs = 1.0 / np.tan(stemp)
    delta = ((F + L)**2.0 - (C - L)**2.0) / (2.0 * C * (C - L))
    epsilon = (A - C) / (2.0 * C)
    gamma = (M - L) / (2.0 * L)
    
    return linerbackus(Vp=vp0, Vs=vs0, Rho=R, Qp=Qp, Qs=Qs, delta=delta, epsilon=epsilon, gamma=gamma)