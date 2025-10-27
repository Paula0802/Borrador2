# plotting.py - save basic plots (matplotlib)
import matplotlib.pyplot as plt
from pathlib import Path
def save_rt_plot(wl, R, T, A, outpath):
    plt.figure(figsize=(7,4))
    plt.plot(wl, R, label='R')
    plt.plot(wl, T, label='T')
    plt.plot(wl, A, label='A')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Fraction')
    plt.legend()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
