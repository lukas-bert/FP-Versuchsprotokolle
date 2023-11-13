import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

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

fig, ax = plt.subplots()
xx = np.linspace(0, np.pi, 1000)
ax.plot(xx, np.sin(xx)*np.cos(xx), label = "sin(x)*cos(x)")
ax.plot(xx, np.abs(np.sin(xx)*np.cos(xx)), label = "Abs")
ax.set_xlabel("Angle")
ax.set_ylabel("Contrast")
ax.legend()

plt.tight_layout()
plt.savefig("build/plot.pdf")
