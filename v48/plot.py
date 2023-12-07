import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)
from scipy.integrate import simps

#Fits
def lin(x, a, b):
    return a*x +b



def exp(x, A_0, b, c):
    return A_0*np.e**(b*x) + c

def linfit(x_data, y_data):
    params, pcov = op.curve_fit(lin, x_data, y_data)
    err = np.sqrt(np.diag(pcov))

    a = ufloat(params[0], err[0])
    b = ufloat(params[1], err[1])
    return a, b


def expfit(x_data, y_data, p0=[1,1,1]):
    params, pcov = op.curve_fit(exp, x_data, y_data, p0)
    err = np.sqrt(np.diag(pcov))

    a = ufloat(params[0], err[0])
    b = ufloat(params[1], err[1])
    c = ufloat(params[2], err[2])
    return a, b, c


T_1, I_1_ = np.genfromtxt("content/data/T_I_1.txt", unpack = True)
T_2, I_2_ = np.genfromtxt("content/data/T_I_2.txt", unpack = True)

T_1 = T_1 + 273.15
T_2 = T_2 + 273.15

I_1 = I_1_*1e-12
I_2 = I_2_*1e-12


#Heizraten berechnen, delta T = 1min
b_1 = np.zeros(len(T_1))
for i in range(len(b_1)-1):
    b_1[i+1] = np.around(T_1[i+1]-T_1[i],2)

b_1_mean = ufloat(np.mean(b_1), np.std(b_1))

b_2 = np.zeros(len(T_2))
for i in range(len(b_2)-1):
    b_2[i+1] = np.around(T_2[i+1]-T_2[i],2)

b_2_mean = ufloat(np.mean(b_2), np.std(b_2))

#select Data for background fit, perform fits
T_1_select = np.concatenate([T_1[2:14], T_1[39:45]]) # T_1[-5:]])#
I_1_select = np.concatenate([I_1_[2:14], I_1_[39:45]]) # I_1_[-5:]])# 

T_2_select = np.concatenate([T_2[2:14], T_2[47:52]])
I_2_select = np.concatenate([I_2_[2:14], I_2_[47:52]])

a_1_fit, b_1_fit, c_1_fit = expfit(T_1_select, I_1_select, p0=[0,0,0])
a_2_fit, b_2_fit, c_2_fit = expfit(T_2_select, I_2_select, p0=[0,0,0])

#subtract background from data
I_1_clean = I_1_ - exp(T_1, noms(a_1_fit), noms(b_1_fit), noms(c_1_fit))
I_2_clean = I_2_ - exp(T_2, noms(a_2_fit), noms(b_2_fit), noms(c_2_fit))

#select data for linear fit, perform linfit
T_1_linfit = T_1[15:31]
I_1_linfit = I_1_clean[15:31]

T_2_linfit = T_2[22:38]
I_2_linfit = I_2_clean[22:38]

m_1_linfit, b_1_linfit = linfit(1/T_1_linfit, np.log(I_1_linfit))
m_2_linfit, b_2_linfit = linfit(1/T_2_linfit, np.log(I_2_linfit))

W_1_linfit = -m_1_linfit*const.k/const.e
W_2_linfit = -m_2_linfit*const.k/const.e

#select data for integral fit
T_1_integral = T_1[31:38]
I_1_integral = I_1_clean[31:38]

T_2_integral = T_2[38:46]
I_2_integral = I_2_clean[38:46]


int1 = []
for  i  in range(len(T_1_integral)-1):
 sim = simps(I_1_integral[i:], T_1_integral[i:])
 int1 = np.append(int1, unp.log(sim/(b_1_mean*I_1_integral[i])))

int2 = []
for  i  in range(len(T_2_integral)-1):
 sim = simps(I_2_integral[i:], T_2_integral[i:])
 int2 = np.append(int2, unp.log(sim/(b_2_mean*I_2_integral[i])))

m_1_integral, b_1_integral = linfit(1/T_1_integral[:-1], noms(int1))
m_2_integral, b_2_integral = linfit(1/T_2_integral[:-1], noms(int2))

W_1_integral = m_1_integral*const.k/const.e
W_2_integral = m_2_integral*const.k/const.e

# calculate taus
def tau(T_max, b, W):
   return T_max**2*const.k / (b*W)

tau_1_lin_max = tau(T_1[np.argmax(I_1_clean)], b_1_mean, W_1_integral)
tau_2_lin_max = tau(T_2[np.argmax(I_2_clean)], b_2_mean, W_2_integral)

tau_1_integral_max = np.e**(b_1_integral)
tau_2_integral_max = np.e**(b_2_integral)


################################################################################################################################################
#Plots
#1. Messreihe roh und fit
fig, ax = plt.subplots()

ax.plot(T_1,I_1_, "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Messdaten")
ax.plot(T_1_select, I_1_select, "1", c="firebrick", markersize=14, markeredgewidth=1.4, label="Daten Hintergrund-Fit")

xx = np.linspace(200, 300, 10000)
ax.plot(xx, exp(xx, noms(a_1_fit), noms(b_1_fit), noms(c_1_fit)), label="Fit")

ax.set_xlabel(r"$T \mathbin{/} \unit{\kelvin}$")
ax.set_ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")

ax.legend()

plt.tight_layout()
plt.savefig("build/T_I_1.pdf")

#2. Messreihe roh und fit
fig, ax = plt.subplots()

ax.plot(T_2,I_2_, "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Messdaten")
ax.plot(T_2_select, I_2_select, "1", c="firebrick", markersize=14, markeredgewidth=1.4, label="Daten Hintergrund-Fit")

xx = np.linspace(200, 300, 10000)
ax.plot(xx, exp(xx, noms(a_2_fit), noms(b_2_fit), noms(c_2_fit)), label="Fit")

ax.set_xlabel(r"$T \mathbin{/} \unit{\kelvin}$")
ax.set_ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")

ax.legend()

plt.tight_layout()
plt.savefig("build/T_I_2.pdf")

#1. Messreihe clean
fig, ax = plt.subplots()

ax.plot(T_1, I_1_clean,  "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Messdaten bereinigt")
ax.plot(T_1_linfit, I_1_linfit, "1", c="firebrick", markersize=14, markeredgewidth=1.4, label="Daten Anstieg")
ax.plot(T_1_integral, I_1_integral, "1", c="darkorange", markersize=14, markeredgewidth=1.4, label="Daten Integral")
ax.set_xlabel(r"$T \mathbin{/} \unit{\kelvin}$")
ax.set_ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")

ax.legend()

plt.tight_layout()
plt.savefig("build/T_I_1_clean.pdf")

#2. Messreihe clean
fig, ax = plt.subplots()

ax.plot(T_2, I_2_clean,  "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Messdaten bereinigt")
ax.plot(T_2_linfit, I_2_linfit, "1", c="firebrick", markersize=14, markeredgewidth=1.4, label="Daten Anstieg")
ax.plot(T_2_integral, I_2_integral, "1", c="darkorange", markersize=14, markeredgewidth=1.4, label="Daten Integral")
ax.set_xlabel(r"$T \mathbin{/} \unit{\kelvin}$")
ax.set_ylabel(r"$I \mathbin{/} \unit{\pico\ampere}$")

ax.legend()

plt.tight_layout()
plt.savefig("build/T_I_2_clean.pdf")

#plots of linfit 1. Messreihe
fig, ax = plt.subplots()

ax.plot(1/T_1_linfit, np.log(I_1_linfit),  "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Daten Anstieg")

xx = np.linspace(0.0038, 0.00435, 1000)
ax.plot(xx, lin(xx, noms(m_1_linfit), noms(b_1_linfit)), c="firebrick", label="Fit")

ax.set_xlabel(r"$1/T \mathbin{/} \unit{\per\kelvin}$")
ax.set_ylabel(r"$\ln(I \mathbin{/} \unit{\pico\ampere}$)")

ax.legend()

plt.tight_layout()
plt.savefig("build/T_I_1_linfit.pdf")


#2. Messreihe
fig, ax = plt.subplots()

ax.plot(1/T_2_linfit, np.log(I_2_linfit),  "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Daten Anstieg")

xx = np.linspace(0.0038, 0.00418, 1000)
ax.plot(xx, lin(xx, noms(m_2_linfit), noms(b_2_linfit)), c="firebrick", label="Fit")

ax.set_xlabel(r"$1/T \mathbin{/} \unit{\per\kelvin}$")
ax.set_ylabel(r"$\ln(I \mathbin{/} \unit{\pico\ampere}$)")

ax.legend()

plt.tight_layout()
plt.savefig("build/T_I_2_linfit.pdf")

#plots for integral fit
fig, ax = plt.subplots()

ax.plot(1/T_1_integral[:-1], noms(int1), "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Daten Integral")

xx = np.linspace(1/T_1_integral[0]+2.5e-5, 1/T_1_integral[-1]-1e-5, 1000)
ax.plot(xx, lin(xx, noms(m_1_integral), noms(b_1_integral)), c="firebrick", label="Fit")

ax.legend()

ax.set_xlabel(r"$1/T \mathbin{/} \unit{\per\kelvin}$")
ax.set_ylabel(r"$\ln \left(\frac{\int_T^\infty i(T')\symup{d}T'}{i(T)\tau_0 b}\right)$")

plt.tight_layout()
plt.savefig("build/int_1.pdf")

# 2. plot
fig, ax = plt.subplots()

ax.plot(1/T_2_integral[:-1], noms(int2), "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Daten Integral")

xx = np.linspace(1/T_2_integral[0]+2.5e-5, 1/T_2_integral[-1]-1e-5, 1000)
ax.plot(xx, lin(xx, noms(m_2_integral), noms(b_2_integral)), c="firebrick", label="Fit")

ax.legend()

ax.set_xlabel(r"$1/T \mathbin{/} \unit{\per\kelvin}$")
ax.set_ylabel(r"$\ln \left(\frac{\int_T^\infty i(T')\symup{d}T'}{i(T)\tau_0 b}\right)$")

plt.tight_layout()
plt.savefig("build/int_2.pdf")


#Plot taus
def tau(T, tau_0, W):
   return tau_0*np.e**(W*const.e/(const.k*T))


fig, ax = plt.subplots()

xx = np.linspace(200, 320, 10000)

ax.plot(T_1, tau(T_1, noms(tau_1_lin_max), noms(W_1_linfit)), label=r"$\tau_{\text{linfit}, 1}$")
ax.plot(T_2, tau(T_2, noms(tau_2_lin_max), noms(W_2_linfit)), label=r"$\tau_{\text{linfit}, 2}$")

#ax.plot(T_1, tau(T_1, noms(tau_1_integral_max), noms(W_1_integral)), label=r"$\tau_{\text{intfit}, 1}$")
#ax.plot(T_2, tau(T_2, noms(tau_2_integral_max), noms(W_2_integral)), label=r"$\tau_{\text{intfit}, 2}$", ylim=[0,])

ax.set_ylim(-0.01,0.1)

ax.legend()

ax.set_xlabel(r"$T \mathbin{/} \unit{\kelvin}$")
ax.set_ylabel(r"$\tau \mathbin{/} \unit{\second}$")

plt.tight_layout()
plt.savefig("build/tau.pdf")

# PRINT
print("#################### V21 ####################")
print("---------------------------------------------")
print("Tabelle Messung 1: t / min; T / °C; T / K; I/ pA; b / K / min")
for i in range(len(b_1)): print(i, "\t", np.around(T_1[i]-273.15), "\t", np.around(T_1[i]), "\t", I_1_[i], "\t", b_1[i])
print(".............................................")
print("Tabelle Messung 2: t / min; T / °C; I/ pA; b / K / min")
for i in range(len(b_2)): print(i, "\t", np.around(T_2[i]-273.15), "\t", np.around(T_2[i]), "\t", I_2_[i], "\t", b_2[i])
print("---------------------------------------------")
print(f"b_1_mean = {b_1_mean}")
print(f"b_2_mean = {b_2_mean}")
print("---------------------------------------------")
print("Paramterer BG-Fit 1")
print(f"a_1_fit = {a_1_fit}")
print(f"b_1_fit = {b_1_fit}")
print(f"c_1_fit = {c_1_fit}")
print(".............................................")
print("Paramterer BG-Fit 2")
print(f"a_2_fit = {a_2_fit}")
print(f"b_2_fit = {b_2_fit}")
print(f"c_2_fit = {c_2_fit}")
print("---------------------------------------------")
print("Parameter Linfit 1")
print(f"m_1_linfit: {m_1_linfit}")
print(f"b_1_linfit: {b_1_linfit}")
print("")
print(f"--> W_1_linfit = {W_1_linfit} eV")
print(".............................................")
print("Parameter Linfit 2")
print(f"m_2_linfit: {m_2_linfit}")
print(f"b_2_linfit: {b_2_linfit}")
print("")
print(f"--> W_2_linfit = {W_2_linfit} eV")
print("---------------------------------------------")
print("Parameter Intergral Linfit 1")
print(f"m_1_integral = {m_1_integral}")
print(f"b_1_integral = {b_1_integral}")
print("")
print(f"--> W_1_integral = {W_1_integral} eV")
print(".............................................")
print("Parameter Intergral Linfit 2")
print(f"m_2_integral = {m_2_integral}")
print(f"b_2_integral = {b_2_integral}")
print("")
print(f"--> W_2_integral = {W_2_integral} eV")
print("---------------------------------------------")
print(f"tau_1_lin_max = {tau_1_lin_max}")
print(f"tau_2_lin_max = {tau_2_lin_max}")
print(f"tau_1_integral_0 = {tau_1_integral_max}")
print(f"tau_2_integral_0 = {tau_2_integral_max}")
print("---------------------------------------------")
print("#################### V21 ####################")