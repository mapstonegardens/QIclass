import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math

"""
rhof2 dan kf2 dapat dimodelkan sebagai effective density dan modulus:

rhofl2=(f1mix*rhof1)+(f2mix*rhof2) # eff density
kreuss=1/((f1mix/kf1)+(f2mix/kf2)) {eff modulus (Domenico, 1976)}
kvoigt=(f1mix*kf1)+(f2mix*kf2) {patchy approach (Mavko and Mukerji, 1998)}

"""

def fluidsub(vp, vs, rho, phi, rhof1, rhof2, kmin, kf1, kf2):
    # hitung bulk dan shear moduli
    musat1 = rho*(vs**2)
    kfl1=rho*((vp**2)-((4/3)*musat1))
    # kalkulasi dry from ksat1 
    upper_kdry=kfl1*((phi*kmin)/kf1+(1-phi))-kmin
    lower_kdry=((phi*kmin)/kfl1)+(kf1/kmin)-1-phi
    kdry=upper_kdry/lower_kdry          
    #replacement------------------------
    upper_k2=(1-(kdry/kmin))**2
    lower_k2=(phi/kf2)+((1-phi)/kmin)-(kdry/(kmin**2))  
    kfl2=kdry+(upper_k2/lower_k2)
    # asumsi shear modulus tidak berubah
    musat2 = musat1
    # hitung ulang densitas karena perubahan fluida
    rho2 = rho + phi*(rhof2 - rhof1)
    # hitung vp2 dan vs2
    vp2 = np.sqrt((kfl2 + (4/3)*musat2) / rho2)
    vs2 = np.sqrt((musat2/rho2))
    return vp2, vs2, rho2