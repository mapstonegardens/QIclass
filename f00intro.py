import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

def crossplot (jumlah_sampel, velocity, porosity):
    surface = jumlah_sampel
    Vp0 = velocity
    Phi = porosity
    Interval = np.zeros(surface + 1)
    declining = np.arange(surface + 1)
    for i in range(surface + 1):
        Interval[i] = Vp0*(1 + Phi)**i
    plt.plot(declining, Interval, "ro", ls="-")
    plt.xlabel("Decline"); plt.ylabel("Interval")
    plt.grid(); plt.gca().invert_yaxis()
