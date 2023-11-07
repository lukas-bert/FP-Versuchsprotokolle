import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

mu_B = 9.2740100783e-24

def B(I,N,R):
    return const.mu_0*(8*I*N)/(np.sqrt(125)*R)

#Data 1
# f/kHz, 1. Peak, 2. Peak, B_hor_peak1 / mV, B_hor_peak2
f_kHz, B_sweep_peak1_arb, B_sweep_peak2_arb, B_hor_peak1_arb, B_hor_peak2_arb = np.genfromtxt("content/data/data1.txt", unpack=True)

f = f_kHz*1e3

B_sweep_peak1 = B(B_sweep_peak1_arb*0.1, 11, 16.39e-2)
B_sweep_peak2 = B(B_sweep_peak2_arb*0.1, 11, 16.39e-2)

B_hor_peak1 = B(B_hor_peak1_arb*2e-3, 154, 15.79e-2)
B_hor_peak2 = B(B_hor_peak2_arb*2e-3, 154, 15.79e-2)

B_peak1 = B_sweep_peak1 + B_hor_peak1 #-B_sweep_peak1
B_peak2 = B_sweep_peak2 + B_hor_peak2 #-B_sweep_peak2

#Data 2
# A/V, delta t_peak1/ms, n_peaks1, delta t_peak2/ms, n_peaks2
A, t_1_ms, n_1, t_2_ms, n_2 = np.genfromtxt("content/data/data2.txt", unpack=True)

t_1 = t_1_ms*1e-3
t_2 = t_2_ms*1e-3

T_1 = t_1/n_1
T_2 = t_2/n_2

#Data 3
# t_peak1/ms, U_peak1/V, t_peak2/ms, U_peak2/V
t_exp_1_ms, A_exp_1, t_exp_2_ms, A_exp_2 = np.genfromtxt("content/data/data3.txt", unpack=True)

t_exp_1 = t_exp_1_ms*1e-3
t_exp_2 = t_exp_2_ms*1e-3

# Calculation B_vert
B_vert = B(0.23, 20, 11.735e-2)


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

# Fit g_F
a_peak1_unc, b_peak1_unc = linfit(f, B_peak1)
a_peak2_unc, b_peak2_unc = linfit(f, B_peak2)

a_peak1 = noms(a_peak1_unc)
b_peak1 = noms(b_peak1_unc)
a_peak2 = noms(a_peak2_unc) 
b_peak2 = noms(b_peak2_unc)


# Fit Rabi
a_R_1, b_R_1, c_R_1 = hyperfit(A, T_1)
a_R_2, b_R_2, c_R_2 = hyperfit(A, T_2)

# Fit Exp. Anstieg
a_max_exp_1,a_min_exp_1, b_exp_1 = expfit(t_exp_1, A_exp_1)
a_max_exp_2,a_min_exp_2, b_exp_2 = expfit(t_exp_2, A_exp_2)

#Calc g_F
def g_F(a):
    return const.h/(a*mu_B)

#Kernspin
J = 0.5
S = 0.5
L = 0

g_J = 1+(J*(J+1)+S*(S+1)-L*(L+1))/(2*J*(J+1))
I1 = g_J / (4 * g_F(a_peak1_unc)) - 1 + unp.sqrt((g_J / (4 * g_F(a_peak1_unc)) - 1)**2+ 3 * g_J / (4 * g_F(a_peak1_unc)) - 3 / 4)
I2 = g_J / (4 * g_F(a_peak2_unc)) - 1 + unp.sqrt((g_J / (4 * g_F(a_peak2_unc)) - 1)**2+ 3 * g_J / (4 * g_F(a_peak2_unc)) - 3 / 4)

#Zeemann^2
U1 = g_F(a_peak1_unc)*mu_B*np.max(B_peak1)+g_F(a_peak1_unc)**2*mu_B**2*np.max(B_peak1)**2*(1-2*2)/(4.53e-24)
U2 = g_F(a_peak2_unc)*mu_B*np.max(B_peak2)+g_F(a_peak2_unc)**2*mu_B**2*np.max(B_peak2)**2*(1-2*3)/(2.01e-24)

#Plots
#Linear
fig,ax = plt.subplots()
ax.plot(f*1e-6,B_peak1*1e6, "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Messdaten 1. Isotop")
ax.plot(f*1e-6,B_peak2*1e6, "1", c="firebrick", markersize=14, markeredgewidth=1.4, label="Messdaten 2. Isotop")

xx = np.linspace(0, 1050*1e3, 1000)
ax.plot(xx*1e-6, lin(xx, a_peak1, b_peak1)*1e6, linestyle="dashed", label="Fit 1. Isotop")
ax.plot(xx*1e-6, lin(xx, a_peak2, b_peak2)*1e6, linestyle="dashed", label="Fit 2. Isotop")

ax.set_xlabel(r"$f \mathbin{/} \unit{\mega\hertz}$")
ax.set_ylabel(r"$B_{\text{hor}} \mathbin{/} \unit{\micro\tesla}$")
ax.legend()

fig.tight_layout()
fig.savefig("build/plot_B_f.pdf")

#Rabi
fig, ax = plt.subplots()
ax.plot(A, T_1*1e3, "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="Periode 1. Isotop")
ax.plot(A, T_2*1e3, "1", c="firebrick", markersize=14, markeredgewidth=1.4, label="Periode 2. Isotop")

xx = np.linspace(0.8,6.5, 1000)
ax.plot(xx, hyper(xx, noms(a_R_1), noms(b_R_1), noms(c_R_1))*1e3, linestyle="dashed", label= "Fit 1. Isotop")
ax.plot(xx, hyper(xx, noms(a_R_2), noms(b_R_2), noms(c_R_2))*1e3, linestyle="dashed", label= "Fit 2. Isotop")
ax.set_xlabel(r"$A \mathbin{/} \unit{\volt}$")
ax.set_ylabel(r"$T \mathbin{/} \unit{\micro\second}$")
ax.legend()

fig.tight_layout()
fig.savefig("build/plot_Rabi.pdf")

#Exp. Anstieg
fig, ax = plt.subplots()
ax.plot(t_exp_1*1e3, A_exp_1, "1", c="cornflowerblue", markersize=14, markeredgewidth=1.4, label="1. Isotop")
ax.plot(t_exp_2*1e3, A_exp_2, "1", c="firebrick", markersize=14, markeredgewidth=1.4, label="2. Isotop")

xx = np.linspace(0,32, 1000)
ax.plot(xx, exp(xx*1e-3, noms(a_max_exp_1), noms(a_min_exp_1), noms(b_exp_1)), linestyle="dashed", label= "Fit 1. Isotop")
ax.plot(xx, exp(xx*1e-3, noms(a_max_exp_2), noms(a_min_exp_2), noms(b_exp_2)), linestyle="dashed", label= "Fit 2. Isotop")

ax.set_xlabel(r"$t \mathbin{/} \unit{\micro\second}$")
ax.set_ylabel(r"$A \mathbin{/} \unit{\volt}$")
ax.legend()

fig.tight_layout()
fig.savefig("build/plot_exp.pdf")


#Print
print("#################### V21 ####################")
print("---------------------------------------------")
print(f"B_vertikal = {B_vert:.3e}")
print("---------------------------------------------")
print(f"a_peak1 = {a_peak1_unc:.2e}")
print(f"b_peak1 = {b_peak1_unc:.2e}")
print("")
print(f"a_peak2 = {a_peak2_unc:.2e}")
print(f"b_peak2 = {b_peak2_unc:.2e}")
print("---------------------------------------------")
print(f"g_F_1 = {g_F(a_peak1_unc)}")
print(f"g_F_2 = {g_F(a_peak2_unc)}")
print(f"g_F_1/g_F_2 = {g_F(a_peak1_unc)/g_F(a_peak2_unc)}")
print("---------------------------------------------")
print(f"I_1 = {I1}")
print(f"I_2 = {I2}")
print("---------------------------------------------")
print("Zeemann^2")
print(f"B_1_max = {np.max(B_peak1):.4e} \t W_1 = {U1} J = {U1/const.e} eV")
print(f"B_2_max = {np.max(B_peak2):.4e} \t W_1 = {U2} J = {U1/const.e} eV")
print("---------------------------------------------")
print("Fitwerte Rabi")
print(f"a_R_1 = {a_R_1}")
print(f"a_R_2 = {a_R_2}")
print(f"b_R_1 = {b_R_1}")
print(f"b_R_2 = {b_R_2}")
print(f"c_R_1 = {c_R_1}")
print(f"c_R_2 = {c_R_2}")
print(f"b2/b1 = {b_R_2/b_R_1}")
print("---------------------------------------------")
print("Fit Exp. Anstieg")
print(f"a_max_exp_1 = {a_max_exp_1}")
print(f"a_max_exp_2 = {a_max_exp_2}")
print(f"a_min_exp_1 = {a_min_exp_1}")
print(f"a_min_exp_2 = {a_min_exp_2}")
print(f"b_exp_1 = {b_exp_1}")
print(f"b_exp_2 = {b_exp_2}")
print("#################### V21 ####################")
