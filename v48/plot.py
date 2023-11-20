import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

T_1, I_1_ = np.genfromtxt("content/data/T_I_1.txt", unpack = True)
T_2, I_2_ = np.genfromtxt("content/data/T_I_2.txt", unpack = True)

I_1 = I_1_*1e-12
I_2 = I_2_*1e-12

#Plots
#1. Messreihe
fig, ax = plt.subplots()

ax.plot(T_1,I_1_*1e12, "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Messdaten")

ax.set_xlabel(r"$T \mathbin{/} \unit{\celsius}$")
ax.set_ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")

ax.legend()

plt.tight_layout()
plt.savefig("build/T_I_1.pdf")

#2. Messreihe
fig, ax = plt.subplots()

ax.plot(T_2,I_2_*1e12, "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Messdaten")

ax.set_xlabel(r"$T \mathbin{/} \unit{\celsius}$")
ax.set_ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")

ax.legend()

plt.tight_layout()
plt.savefig("build/T_I_2.pdf")

#Fits
def lin(x, a, b):
    return a*x +b

def hyper(x,a,b,c):
    if x.any()==c.any():
        c-=0.0000000001
    return a + b/(x-c)

def exp(x, A_max, A_min, b):
    return A_max-(A_max-A_min)*np.e**(-b*x)

def linfit(x_data, y_data):
    params, pcov = op.curve_fit(lin, x_data, y_data)
    err = np.sqrt(np.diag(pcov))

    a = ufloat(params[0], err[0])
    b = ufloat(params[1], err[1])
    return a, b

def hyperfit(x_data, y_data):
    params, pcov = op.curve_fit(hyper, x_data, y_data)
    err = np.sqrt(np.diag(pcov))

    a = ufloat(params[0], err[0])
    b = ufloat(params[1], err[1])
    c = ufloat(params[2], err[2])
    return a, b, c

def expfit(x_data, y_data):
    params, pcov = op.curve_fit(exp, x_data, y_data)
    err = np.sqrt(np.diag(pcov))

    a = ufloat(params[0], err[0])
    b = ufloat(params[1], err[1])
    c = ufloat(params[2], err[2])
    return a, b, c
