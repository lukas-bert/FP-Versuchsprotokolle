import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

lamda = 632.99e-9 # wavelength of the laser (meter)

# Determination of maximum contrast

theta, I_min1, I_max1, I_min2, I_max2, I_min3, I_max3 = np.genfromtxt("content/data/contrast.txt", unpack = True)

theta = theta*np.pi/180

def contrast(I_min, I_max):
    return (I_max - I_min)/(I_max + I_min)

K1 = contrast(I_min1, I_max1)
K2 = contrast(I_min2, I_max2)
K3 = contrast(I_min3, I_max3)

K_mean = np.mean([K1, K2, K3], axis = 0)
K_std = np.std([K1, K2, K3], axis = 0)

def theo_curve(phi, I_0, delta):
    return I_0 *2*np.abs(np.cos(phi - delta)*np.sin(phi - delta))

params, pcov = op.curve_fit(theo_curve, theta, K_mean, p0 = [1, 0])
err = np.sqrt(np.diag(pcov))

print("--------------------------------------------------")
print("Contrast-fit:")
print(f"I_O = {params[0]:.4f} +- {err[0]:.4f}")
print(f"delta / ° = {180*params[1]/np.pi:.4f} +- {180*err[1]/np.pi:.4f}")
print("--------------------------------------------------")

x = np.linspace(-0.1, np.pi + 0.1, 1000)

fig, ax = plt.subplots()

ax.errorbar(theta, K_mean, yerr = K_std, lw = 0, marker = ".", ms = 8, color = "black", label = "Data", elinewidth=1, capsize=4)
ax.plot(x, theo_curve(x, *params), c = "firebrick", label = "Fit (theory curve)")
plt.xlim(-0.1, np.pi+0.1)
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [0, 45, 90, 135, 180])
plt.ylabel(r"Contrast $K$ [a.u.]")
plt.xlabel(r"$\phi \mathbin{/} °$")
plt.ylim(0, 0.7)
plt.legend()
plt.grid()
plt.tight_layout()
#plt.show()

plt.savefig("build/contrast.pdf")
plt.close()

# Determination of the refractive index of glas

theta, M1, M2, M3, M4, M5, M6, M7, M8, M9, M10 = np.genfromtxt("content/data/glass.txt", unpack = True)

theta = theta*np.pi/180

M_mean = np.mean([M1, M2, M3, M4, M5, M6, M7, M8, M9, M10], axis = 0)
M_std = np.std([M1, M2, M3, M4, M5, M6, M7, M8, M9, M10], axis = 0)

T = 1e-3 # thickness of the glassplate
alpha_0 = 10*np.pi/180

def Maxima(theta, n):
    return 2*T/lamda * (n-1)/n *alpha_0*theta

params1, pcov1 = op.curve_fit(Maxima, theta, M_mean)
err1 = np.sqrt(np.diag(pcov1))

print("--------------------------------------------------")
print("Refractive Index Glass:")
print(f"n = {params1[0]:.4f} +- {err1[0]:.4f}")
print("--------------------------------------------------")

x = np.linspace(0, 10, 100)

fig, ax = plt.subplots()

ax.errorbar(theta*180/np.pi, M_mean, yerr = M_std, lw = 0, marker = ".", ms = 8, color = "black", label = "Data", elinewidth=1, capsize=4)
ax.plot(x, Maxima(x*np.pi/180, *params1), c = "firebrick", label = "Fit")

plt.ylabel(r"Number of maxima $M$")
plt.xlabel(r"$\theta \mathbin{/} °$")
plt.xlim(0, 10)
plt.ylim(0, 35)
plt.legend()
plt.grid()
plt.tight_layout()
#plt.show()

plt.savefig("build/n_glass.pdf")
plt.close()

# Refractive index of air

p, m1, m2, m3, m4, m5 = np.genfromtxt("content/data/air.txt", unpack = True)

T_0 = 273.15 + 22.6 # K Raumtemperatur
p = p
L = ufloat(100e-3, 0.1e-3)

def n_air_exp(M):
    return M*lamda/(L) + 1

M = unp.uarray(np.mean([m1, m2, m3, m4, m5], axis = 0), np.std([m1, m2, m3, m4, m5], axis = 0))

n = n_air_exp(M)
print("---------------------------------------------------")
print("Brechungsindices Luft:")
for i in range(len(n)):
    print(f"{n[i]:.7f}")
print("---------------------------------------------------")

def n_air_theo(p, A, b):
    return 3/2 * A*p/(const.R *T) + b

params2, pcov2 = op.curve_fit(n_air_theo, p, noms(n))
err2 = np.sqrt(np.diag(pcov2))

n_air_exp = 3/2 * ufloat(params2[0], err2[0])*1013/(const.R *(273.15 + 15)) + params2[1]

print("--------------------------------------------------")
print("Fit: Refractive Index of Air:")
print(f"A = {params2[0]:.4f} +- {err2[0]:.4f}")
print(f"b = {params2[1]:.8f} +- {err2[1]:.8f}")
print(f"Experimental Value: n = {n_air_exp}")
print("--------------------------------------------------")

fig, ax = plt.subplots()

x = np.linspace(0, 1000, 100)

ax.errorbar(p, noms(n), yerr = stds(n), lw = 0, marker = ".", ms = 8, color = "black", label = "Data", elinewidth=1, capsize=4)
ax.plot(x, n_air_theo(x, *params2), c = "firebrick", label = "Fit")

plt.ylabel(r"$n$")
plt.xlabel(r"$p \mathbin{/} \unit{\milli\bar}$")

plt.legend()
plt.grid()
plt.tight_layout()
plt.xlim(0, 1000)
plt.ylim(1, 1 + 2.8e-4)
#plt.show()

plt.savefig("build/n_air.pdf")
plt.close()
