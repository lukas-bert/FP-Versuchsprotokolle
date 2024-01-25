import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)
from scipy.signal import find_peaks
from scipy.signal import argrelmax

# -------- Linfit -------
def lin(x, a, b):
    return a*x +b
def linfit(x_data, y_data):
    params, pcov = op.curve_fit(lin, x_data, y_data)
    err = np.sqrt(np.diag(pcov))

    a = ufloat(params[0], err[0])
    b = ufloat(params[1], err[1])
    return a, b
# ------ Functions ------
def f_cut(V, m, b):
    return np.e**((unp.log(V/np.sqrt(2))-b)/m)
# -----------------------
# -----------------------
# ----- Import data -----
f_inv_1, phi_inv_1, U_inv_1 = np.genfromtxt("content/data/invert1.txt", unpack=True) # f in kHz
f_inv_2, phi_inv_2, U_inv_2 = np.genfromtxt("content/data/invert2.txt", unpack=True) # f in kHz
f_inv_3, phi_inv_3, U_inv_3 = np.genfromtxt("content/data/invert3.txt", unpack=True) # f in kHz

f_int, U_int = np.genfromtxt("content/data/integrator.txt", unpack=True) # f in Hz
f_diff, U_diff = np.genfromtxt("content/data/differenzierer.txt", unpack=True) # f in Hz

V_int = U_int/0.2
V_diff = U_diff/0.2

# Data Generator
#t_scope, ramp_scope, signal_scope = np.genfromtxt("content/scope/scope_14.csv", unpack=True, delimiter=",")
t_gen, square_scope, signal_scope = np.genfromtxt("content/scope/scope_18.csv", unpack=True, delimiter=",")
# -----------------------
# ----- Select data -----
cut_1 = 2
f_inv_1_av = f_inv_1[:cut_1]
V_inv_1_av = U_inv_1[:cut_1]/0.2
f_inv_1_fl = f_inv_1[cut_1:]
V_inv_1_fl = U_inv_1[cut_1:]/0.2

cut_2 = 7
f_inv_2_av = f_inv_2[:cut_2]
V_inv_2_av = U_inv_2[:cut_2]/0.2
f_inv_2_fl = f_inv_2[cut_2:]
V_inv_2_fl = U_inv_2[cut_2:]/0.2

cut_3 = 2
f_inv_3_av = f_inv_3[:cut_3]
V_inv_3_av = U_inv_3[:cut_3]/0.2
f_inv_3_fl = f_inv_3[cut_3:]
V_inv_3_fl = U_inv_3[cut_3:]/0.2

t_gen_cut = t_gen[582:1500]
signal_scope_cut = signal_scope[582:1500]
# -----------------------

# - Calculate Fits etc. -
ll_V_1 = ufloat(np.mean(V_inv_1_av), np.std(V_inv_1_av))
m_inv_1_fl, b_inv_1_fl = linfit(np.log(f_inv_1_fl), np.log(V_inv_1_fl))

ll_V_2 = ufloat(np.mean(V_inv_2_av), np.std(V_inv_2_av))
m_inv_2_fl, b_inv_2_fl = linfit(np.log(f_inv_2_fl), np.log(V_inv_2_fl))

ll_V_3 = ufloat(np.mean(V_inv_3_av), np.std(V_inv_3_av))
m_inv_3_fl, b_inv_3_fl = linfit(np.log(f_inv_3_fl), np.log(V_inv_3_fl))

f_cutoff_1 = f_cut(ll_V_1, m_inv_1_fl, b_inv_1_fl)
f_cutoff_2 = f_cut(ll_V_2, m_inv_2_fl, b_inv_2_fl)
f_cutoff_3 = f_cut(ll_V_3, m_inv_3_fl, b_inv_3_fl)

m_int, b_int = linfit(np.log(f_int), np.log(V_int)) # linfit integrator
RC_int = 10e3*100e-9
m_diff, b_diff = linfit(np.log(f_diff), np.log(V_diff)) # linfit differentiator
RC_diff = 100e3*22e-9

peaks1, _ = find_peaks(np.around(signal_scope[582:1500], 5), height=0)
peaks2 = argrelmax(signal_scope_cut[peaks1])
peaks = peaks1[peaks2]
peaks = np.delete(peaks, [3,5,7,9,11,12,13,14])

T_gen_arr = t_gen_cut[peaks]
T_mean_arr = np.zeros(len(T_gen_arr)-1)
for i in range(len(T_gen_arr)-1):
    T_mean_arr[i] = T_gen_arr[i+1]-T_gen_arr[i]
T_gen = ufloat(np.mean(T_mean_arr), np.std(T_mean_arr))

# -----------------------

# -------- Plots --------
# Plot Inv 1 
fig, ax = plt.subplots()

ax.plot(np.log(f_inv_1_av), np.log(V_inv_1_av), "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Messwerte für Mittelwertsberechnung")
ax.plot(np.log(f_inv_1_fl), np.log(V_inv_1_fl), "1", c="firebrick", markersize=14, markeredgewidth=1.4, label="Messwerte für Flankenfit")

ax.hlines(np.log(noms(ll_V_1)), xmin=np.log(f_inv_1_av[0])-0.2, xmax=np.log(f_inv_1_av[-1])+0.2, colors="royalblue")
xx = np.linspace(np.log(f_inv_1_fl[0])-0.2, np.log(f_inv_1_fl[-1])+0.2, 1000)
ax.plot(xx, noms(m_inv_1_fl)*xx+noms(b_inv_1_fl), c="peru")

ax.hlines(np.log(noms(ll_V_1)/np.sqrt(2)), xmin=np.log(f_inv_1_av[0])-0.2, xmax=np.log(f_inv_1_fl[-1])+0.2, linestyles="dashed", color = "gray", label = r"${V_1}\mathbin{/}{\sqrt{2}}$")
ax.vlines(np.log(noms(f_cutoff_1)), ymin=np.log(V_inv_1_fl[-1])-0.2, ymax=np.log(V_inv_1_av[0])+0.1, linestyles="dashed", color = "darkorange", label=r"$f_{\mathrm{cutoff}}$")

ax.set_xlabel(r"$\ln\left(\frac{f}{\unit{\kilo\hertz}}\right)$")
ax.set_ylabel(r"$\ln\left(V = \frac{U_2}{U_1}\right)$")
ax.legend()

fig.tight_layout()
fig.savefig("build/invert1.pdf")

# Plot Inv 2
fig, ax = plt.subplots()

ax.plot(np.log(f_inv_2_av), np.log(V_inv_2_av), "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Messwerte für Mittelwertsberechnung")
ax.plot(np.log(f_inv_2_fl), np.log(V_inv_2_fl), "1", c="firebrick", markersize=14, markeredgewidth=1.4, label="Messwerte für Flankenfit")

ax.hlines(np.log(noms(ll_V_2)), xmin=np.log(f_inv_2_av[0])-0.2, xmax=np.log(f_inv_2_av[-1])+0.2, colors="royalblue")
xx = np.linspace(np.log(f_inv_2_fl[0])-0.2, np.log(f_inv_2_fl[-1])+0.2, 1000)
ax.plot(xx, noms(m_inv_2_fl)*xx+noms(b_inv_2_fl), c="peru")

ax.hlines(np.log(noms(ll_V_2)/np.sqrt(2)), xmin=np.log(f_inv_2_av[0])-0.2, xmax=np.log(f_inv_2_fl[-1])+0.2, linestyles="dashed", color = "gray", label = r"${V_2}\mathbin{/}{\sqrt{2}}$")
ax.vlines(np.log(noms(f_cutoff_2)), ymin=np.log(V_inv_2_fl[-1])-0.2, ymax=np.log(V_inv_2_av[0])+0.1, linestyles="dashed", color = "darkorange", label=r"$f_{\mathrm{cutoff}}$")

ax.set_xlabel(r"$\ln\left(\frac{f}{\unit{\kilo\hertz}}\right)$")
ax.set_ylabel(r"$\ln\left(V = \frac{U_2}{U_1}\right)$")
ax.legend()

fig.tight_layout()
fig.savefig("build/invert2.pdf")

# Plot Inv 3
fig, ax = plt.subplots()

ax.plot(np.log(f_inv_3_av), np.log(V_inv_3_av), "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Messwerte für Mittelwertsberechnung")
ax.plot(np.log(f_inv_3_fl), np.log(V_inv_3_fl), "1", c="firebrick", markersize=14, markeredgewidth=1.4, label="Messwerte für Flankenfit")

ax.hlines(np.log(noms(ll_V_3)), xmin=np.log(f_inv_3_av[0])-0.2, xmax=np.log(f_inv_3_av[-1])+0.2, colors="royalblue")
xx = np.linspace(np.log(f_inv_3_fl[0])-0.2, np.log(f_inv_3_fl[-1])+0.2, 1000)
ax.plot(xx, noms(m_inv_3_fl)*xx+noms(b_inv_3_fl), c="peru")

ax.hlines(np.log(noms(ll_V_3)/np.sqrt(2)), xmin=np.log(f_inv_3_av[0])-0.2, xmax=np.log(f_inv_3_fl[-1])+0.2, linestyles="dashed", color = "gray", label = r"${V_3}\mathbin{/}{\sqrt{2}}$")
ax.vlines(np.log(noms(f_cutoff_3)), ymin=np.log(V_inv_3_fl[-1])-0.2, ymax=np.log(V_inv_3_av[0])+0.1, linestyles="dashed", color = "darkorange", label=r"$f_{\mathrm{cutoff}}$")

ax.set_xlabel(r"$\ln\left(\frac{f}{\unit{\kilo\hertz}}\right)$")
ax.set_ylabel(r"$\ln\left(V = \frac{U_2}{U_1}\right)$")
ax.legend()

fig.tight_layout()
fig.savefig("build/invert3.pdf")

# Plot Phase
fig, ax = plt.subplots()

ax.plot(np.log(f_inv_1), abs(phi_inv_1), "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label=r"$V_{\mathrm{theo}}=100$")
ax.plot(np.log(f_inv_2), abs(phi_inv_2), "1", c="firebrick", markersize=14, markeredgewidth=1.4, label=r"$V_{\mathrm{theo}}=15$")
ax.plot(np.log(f_inv_3), abs(phi_inv_3), "1", c="darkgreen", markersize=14, markeredgewidth=1.4, label=r"$V_{\mathrm{theo}}=179$")

ax.set_xlabel(r"$\ln\left(\frac{f}{\unit{\kilo\hertz}}\right)$")
ax.set_ylabel(r"$|\varphi| \mathrm{/} \unit{\degree}$")
ax.legend()

fig.tight_layout()
fig.savefig("build/invert_phase.pdf")

# Plot Integrator
fig, ax = plt.subplots()

ax.plot(np.log(f_int), np.log(V_int), "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Messwerte")
xx = np.linspace(np.log(f_int[0])-0.2, np.log(f_int[-1])+0.2, 1000)
ax.plot(xx, lin(xx, noms(m_int), noms(b_int)), color="firebrick", label="Fit")
ax.plot(xx, lin(xx, -1, np.log(1/RC_int)), linestyle="dashed", color="gray", label="Theorie")

ax.set_xlabel(r"$\ln\left(\frac{f}{\unit{\hertz}}\right)$")
ax.set_ylabel(r"$\ln\left(V = \frac{U_2}{U_1}\right)$")
ax.legend()

fig.tight_layout()
fig.savefig("build/integrator.pdf")

# Plot Differentiator
fig, ax = plt.subplots()
ax.plot(np.log(f_diff), np.log(V_diff), "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Messwerte")
xx = np.linspace(np.log(f_diff[0])+0.1, np.log(f_diff[-1])-0.1, 1000)
ax.plot(xx, lin(xx, noms(m_diff), noms(b_diff)), color="firebrick", label="Fit")
ax.plot(xx, lin(xx, 1, np.log(RC_diff)), linestyle="dashed", color="gray", label="Theorie")

ax.set_xlabel(r"$\ln\left(\frac{f}{\unit{\kilo\hertz}}\right)$")
ax.set_ylabel(r"$\ln\left(V = \frac{U_2}{U_1}\right)$")
ax.legend()

fig.tight_layout()
fig.savefig("build/differentiator.pdf")

# Plot Generator
fig, ax = plt.subplots()
ax.plot(t_gen, square_scope, c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Input")
ax.plot(t_gen, signal_scope, c="firebrick", markersize=14, markeredgewidth=1.4, label="Signal")

ax.set_xlabel(r"$t \mathbin{/} \unit{\second}$")
ax.set_ylabel(r"$U \mathbin{/} \unit{\volt}$")
ax.legend()

fig.tight_layout()
fig.savefig("build/generator_scope.pdf")

fig, ax = plt.subplots()
ax.plot(t_gen_cut, signal_scope_cut, c="firebrick", markersize=14, markeredgewidth=1.4, label="Signal")
ax.plot(t_gen_cut[peaks], signal_scope_cut[peaks], "x", markersize=14, markeredgewidth=1.4, label="Peaks")

ax.set_xlabel(r"$t \mathbin{/} \unit{\second}$")
ax.set_ylabel(r"$U \mathbin{/} \unit{\volt}$")
ax.legend()

fig.tight_layout()
fig.savefig("build/generator_scope_zoomed.pdf")
# -------- Print --------
print("#################### V51 ####################")
print("Theoriewerte Leerlaufverstärkung")
print(f"ll_V_1_theo = {100}")
print(f"ll_V_2_theo = {15}")
print(f"ll_V_3_theo = {68000/380:2f}")
print("")
print("Mittelwerte Leerlaufverstärkung")
print(f"ll_V_1:{ll_V_1}")
print(f"ll_V_2:{ll_V_2}")
print(f"ll_V_3:{ll_V_3}")
print("")
print("---------------------------------------------")
print("Fitparameter Flankenfit")
print(f"m_inv_1_fl = {m_inv_1_fl}")
print(f"b_inv_1_fl = {b_inv_1_fl}")
print("")
print(f"m_inv_2_fl = {m_inv_2_fl}")
print(f"b_inv_2_fl = {b_inv_2_fl}")
print("")
print(f"m_inv_3_fl = {m_inv_3_fl}")
print(f"b_inv_3_fl = {b_inv_3_fl}")
print("")
print("---------------------------------------------")
print("Grenzfrequenzen")
print(f"f_cutoff_1 = {f_cutoff_1}")
print(f"f_cutoff_2 = {f_cutoff_2}")
print(f"f_cutoff_3 = {f_cutoff_3}")
print("")
print("Bandbreitenprodukt")
print(f"V_1*f_cutoff_1 = {ll_V_1*f_cutoff_1}")
print(f"V_1*f_cutoff_2 = {ll_V_1*f_cutoff_2}")
print(f"V_1*f_cutoff_3 = {ll_V_1*f_cutoff_3}")
print("")
print("---------------------------------------------")
print("Integrator")
print(f"m_int = {m_int}")
print(f"b_int = {b_int}")
print("")
print(f"RC_int = {RC_int}")
print(f"1/RC_int = {1/RC_int}")
print("")
print(f"1/RC_exp = {np.e**b_int}")
print(f"RC_exp = {1/(np.e**b_int)}")
print("")
print("---------------------------------------------")
print("Differentiator")
print(f"m_diff = {m_diff}")
print(f"b_diff = {b_diff}")
print("")
print(f"RC_diff = {RC_diff}")
print("")
print(f"RC_exp = {np.e**b_diff}")
print("")
print("---------------------------------------------")
print("Generator")

def nu_a(R1, R2, R3, C):
    return R2/(4*C*R1*R3)
def U_0(U_max,R1,R2):
    return U_max*R1/R2

print(f"nu_a = {nu_a(10e3,100e3,1e3,1e-6)} Hz")
print(f"U_0 = {U_0(15, 10e3,100e3)}")
print("")
print(f"T_gen_theo 100nF = {2*np.pi*10e3*100e-9}")
print(f"T_gen_theo 22nF = {2*np.pi*10e3*22e-9}")
print(f"T_gen_exp = {T_gen}")
print("#################### V51 ####################")
# -----------------------