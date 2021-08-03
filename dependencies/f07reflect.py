import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math

def fullzoep(vp1, vs1, rho1, vp2, vs2, rho2, theta):
    vp1  = float(vp1); vp2  = float(vp2)
    vs1  = float(vs1); vs2  = float(vs2)
    rho1 = float(rho1); rho2 = float(rho2)
    theta1 = float(theta); thetarad = math.radians(theta1)
    p = math.sin(math.radians(thetarad))/vp1; theta2 = math.asin(p*vp2)
    phi1   = math.asin(p*vs1); phi2   = math.asin(p*vs2)
    A = np.array([ \
        [-math.sin(thetarad), -math.cos(phi1), math.sin(theta2), math.cos(phi2)],\
        [math.cos(thetarad), -math.sin(phi1), math.cos(theta2), -math.sin(phi2)],\
        [2*rho1*vs1*math.sin(phi1)*math.cos(thetarad), rho1*vs1*(1-2*math.sin(phi1)**2),\
            2*rho2*vs2*math.sin(phi2)*math.cos(theta2), rho2*vs2*(1-2*math.sin(phi2)**2)],\
        [-rho1*vp1*(1-2*math.sin(phi1)**2), rho1*vs1*math.sin(2*phi1), \
            rho2*vp2*(1-2*math.sin(phi2)**2), -rho2*vs2*math.sin(2*phi2)]
        ], dtype='float')
    B = np.array([ \
        [math.sin(thetarad), math.cos(phi1), -math.sin(theta2), -math.cos(phi2)],\
        [math.cos(thetarad), -math.sin(phi1), math.cos(theta2), -math.sin(phi2)],\
        [2*rho1*vs1*math.sin(phi1)*math.cos(thetarad), rho1*vs1*(1-2*math.sin(phi1)**2),\
            2*rho2*vs2*math.sin(phi2)*math.cos(theta2), rho2*vs2*(1-2*math.sin(phi2)**2)],\
        [rho1*vp1*(1-2*math.sin(phi1)**2), -rho1*vs1*math.sin(2*phi1),\
            -rho2*vp2*(1-2*math.sin(phi2)**2), rho2*vs2*math.sin(2*phi2)]\
        ], dtype='float') 
    Refl = np.dot(np.linalg.inv(A), B);
    return Refl

def ei(vp, vs, rho, alpha, k=0.25):
    alpha = np.radians(alpha)
    #q = np.tan(alpha)**2
    q = np.sin(alpha)**2
    w = -8*k * (np.sin(alpha)**2)
    e = 1 - 4*k * (np.sin(alpha)**2)
    rho_star = vp**q *  vs**w * rho**e
    ei = vp * rho_star
    return ei

def ei_norm(vp, vs, rho, alpha, scal, k=0.25):
    alpha = np.radians(alpha)
    vp0, vs0, rho0 = scal[0], scal[1], scal[2]
    a = 1 + (np.tan(alpha)) ** 2
    b = -8 * k * ((np.sin(alpha)) ** 2)
    c = 1 - 4 * k * ((np.sin(alpha)) ** 2)
    ei = vp0*rho0 * ( (vp/vp0) ** a * (vs/vs0) ** b * (rho/rho0) ** c)
    return ei