import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds

###########################################################################################################################
# Stability of the Laser

r1 = np.inf
r2 = 1 # meter
r3 = 1.4 # meter
  
def g(L, r):
    return 1 - L/r

fig1, ax = plt.subplots()

l = np.linspace(0, 3.0, 10000)

ax.hlines([0, 1], 0, 3.0, ls = "dotted", colors = "grey")
#ax.plot(l, g(l, r1)*g(l, r1), label = "flat / flat")
#ax.plot(l, g(l, r2)*g(l, r1), label = r"$r = 1000 \unit{\milli\metre} / r = \infty$")
#ax.plot(l, g(l, r2)*g(l, r2), label = r"$r_1 = \qty{1400}{\milli\metre}, r_2 = \qty{1000}{\milli\metre}$", ls = "dashed", lw = 1)
ax.plot(l, g(l, r3)*g(l, r1), label = r"$r_1 = \qty{1400}{\milli\metre}, r_2 = \infty$")
ax.plot(l, g(l, r3)*g(l, r3), label = r"$r_1 = \qty{1400}{\milli\metre}, r_2 = \qty{1400}{\milli\metre}$")

ax.plot(1.4, 0, marker = "x", lw = 0, c = "black", ms = 10)
ax.plot(2.8, 1, marker = "x", lw = 0, c = "black", ms = 10)

ax.legend()
ax.set_ylabel(r"$g_1 g_2$")
ax.set_xlabel(r"$L \mathbin{/} \unit{\metre}$")
ax.set_xlim(0, 3.0)
ax.set_ylim(-1.5, 2.5)
ax.grid()
plt.tight_layout()

#fig1.show()

fig1.savefig("build/stability.pdf")
plt.close()

###########################################################################################################################
# Polarisation

theta, I = np.genfromtxt("content/data/Polarisation.txt", unpack = True)
theta = theta/180*np.pi # degree to rad

def f(theta, I_1, delta, I_0):
    return I_1*np.sin(theta + delta)**2 + I_0

params, pcov = op.curve_fit(f, theta, I)
err = np.sqrt(np.diag(pcov))

print("---------------------------------------------------------")
print("Fitparamter Polarisationsfit:")
print(r"I_1:    ", f"{params[0]:.4f} +- {err[0]:.4f}" )
print(r"delta:  ", f"{params[1]*180/np.pi:.4f} +- {err[1]*180/np.pi:.4f}" )
print(r"I_0:    ", f"{params[2]:.4e} +- {err[2]:.4e}" )
print("---------------------------------------------------------")


x = np.linspace(-0.2, 2*np.pi+0.2, 10000)

plt.plot(theta, I, label = "Messwerte", marker = "x", color = "firebrick", lw = 0)
plt.plot(x, f(x, *params), c = "cornflowerblue", label = "Fit")

plt.xlabel(r"$\theta \mathbin{/} \unit{\radian}$")
plt.ylabel(r"$I \mathbin{/} \unit{\milli\watt}$")
plt.xlim(-0.2, 2*np.pi+0.2)
plt.ylim(0, 3.5)
plt.xticks([0, np.pi/2, np.pi, 3/2*np.pi, 2*np.pi], [0, r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])

plt.legend()
plt.grid()
plt.tight_layout()
#plt.show()

plt.savefig("build/polarisation.pdf")
plt.close()

###########################################################################################################################
# Transversal Modes (TEM) Plots

d, I_00, I_01 = np.genfromtxt("content/data/TEM.txt", unpack = True)

# TEM_00

def f00(x, I_0, x_0, w):
    return I_0*np.exp(-(x-x_0)**2/(w**2))

params00, pcov00 = op.curve_fit(f00, d, I_00)
err00 = np.sqrt(np.diag(pcov00))

print("---------------------------------------------------------")
print("Fitparamter TEM 00:")
print(r"I_0:    ", f"{params00[0]:.4f} +- {err00[0]:.4f}" )
print(r"x_0:    ", f"{params00[1]:.4f} +- {err00[1]:.4f}" )
print(r"w:      ", f"{params00[2]:.4f} +- {err00[2]:.4f}" )
print("---------------------------------------------------------")

x = np.linspace(-25, 25, 10000)

plt.plot(d, I_00, label = "Messwerte", marker = "x", color = "firebrick", lw = 0)
plt.plot(x, f00(x, *params00), label = "Fit", c = "cornflowerblue")
plt.xlabel(r"$d \mathbin{/} \unit{\milli\metre}$")
plt.ylabel(r"$I \mathbin{/} \unit{\micro\ampere}$")

plt.xlim(-22.5, 22.5)
#plt.ylim(0, 4.5)

plt.legend()
plt.grid()
plt.tight_layout()
#plt.show()

plt.savefig("build/TEM00.pdf")
plt.close()

# TEM_01

def f01(x, I_0, x_0, w):
    return I_0*8*(x-x_0)**2/(w**2)*np.exp(-(x-x_0)**2/(w**2))

params01, pcov01 = op.curve_fit(f01, d, I_01)
err01 = np.sqrt(np.diag(pcov01))

print("---------------------------------------------------------")
print("Fitparamter TEM 01:")
print(r"I_0:    ", f"{params01[0]:.4f} +- {err01[0]:.4f}" )
print(r"x_0:    ", f"{params01[1]:.4f} +- {err01[1]:.4f}" )
print(r"w:      ", f"{params01[2]:.4f} +- {err01[2]:.4f}" )
print("---------------------------------------------------------")

plt.plot(d, I_01, label = "Messwerte", marker = "x", color = "firebrick", lw = 0)
plt.plot(x, f01(x, *params01), label = "Fit", c = "cornflowerblue")
plt.xlabel(r"$d \mathbin{/} \unit{\milli\metre}$")
plt.ylabel(r"$I \mathbin{/} \unit{\micro\ampere}$")

plt.xlim(-22.5, 22.5)

plt.legend()
plt.grid()
plt.tight_layout()
#plt.show()

plt.savefig("build/TEM01.pdf")
plt.close()

###########################################################################################################################
# Multimode operation

f_0 = 1.5e9 # Hertz
T_0 = 293.15 # K
m = 20.1797*const.u

f = np.sqrt(8*const.k*T_0*np.log(2)/(m*const.c**2))*f_0

print("---------------------------------------------------------")
print("Dopplerverbreiterung:")
print(f"delta f = {f:.4f}" )
print("---------------------------------------------------------")

# Measurements
L = [50, 75, 100, 125, 150, 175, 200]

freqs = [
    [304, 611, 919],
    [203, 405, 604, 806, 1009],
    [150, 300, 454, 600, 754, 904, 1054],
    [124, 240, 364, 480, 600, 720, 840, 960, 1080, 1204],
    [101, 203, 304, 401, 503, 604, 701, 803, 904, 1005, 1106, 1208],
    [86, 176, 260, 350, 435, 518, 600, 686, 773, 863, 949, 1031, 1121],
    [75, 154, 221, 300, 375, 450, 525, 596, 670, 754, 825, 904, 980, 1054]
]

mean = []
std = []

for i in range(len(L)):
    mean.append(np.mean(np.diff(freqs[i])))
    std.append(np.std(np.diff(freqs[i])))

delta_f=unp.uarray(mean, std)

L_exp = const.c/(2*delta_f*10**6)

plt.errorbar(np.array(L), noms(1/delta_f), yerr = stds(1/delta_f), label = "Messwerte", marker = ".", color = "black", lw = 0, capsize= 2, elinewidth= 1)
l = np.linspace(50, 200, 100)
plt.plot(l, 2/const.c*10**(4)*l, label = "Theoriekurve", c = "firebrick")
plt.xlabel(r"$L \mathbin{/} \unit{\centi\metre}$")
plt.ylabel(r"$\frac{1}{\symup{\Delta}f} \mathbin{/} \unit{\mega\hertz^{-1}}$")

plt.legend()
plt.grid()
plt.tight_layout()
#plt.show()

plt.savefig("build/multimode.pdf")
plt.close()

print("---------------------------------------------------------")
print("Multimoden Frequenzspektrum:")
for i in range(len(L)):
    print(f"L = {L[i]} cm,     delta_f = {delta_f[i]:.4f} MHz,     L_exp = {L_exp[i]*100:.2f} cm")
print("---------------------------------------------------------")

###########################################################################################################################
# Wavelength of the laser

def lamda(g, d, d_ii):
    l = np.zeros(len(d_ii))
    for i in range(len(l)):
        l[i] = 1/(g*(10**3)*(i+1)) * np.sin(np.arctan(d_ii[i]/(d*2)))
    return l

g1 = 1200 # /mm
d1 = 25 # cm
d_ii1 = [58] # cm 

g2 = 600 # /mm
d2 = 25 # cm
d_ii2 = [20.5, 59.5] # cm 

g3 = 100 # /mm
d3 = 80 # cm
d_ii3 = [10, 20.5, 31, 42] # cm

g4 = 80 # /mm
d4 = 110 # cm
d_ii4 = [11.5, 22.5, 33.5] # cm

l = np.concatenate((lamda(g1, d1, d_ii1), lamda(g2, d2, d_ii2), lamda(g3, d3, d_ii3), lamda(g4, d4, d_ii4)))

print("---------------------------------------------------------")
print("Wellenlänge:")
for i in l:
    print(f"{i:.5e}")
print("Mittelwert = ", f"{np.mean(l):.5e}", " +- ", f"{np.std(l):.5e}")
print("---------------------------------------------------------")
