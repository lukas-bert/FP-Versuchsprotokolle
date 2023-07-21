import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as devs
from uncertainties import ufloat
import scipy.optimize as op

t, c1 = np.genfromtxt("content/data/ReflectScan.UXD", unpack=True)
t2, c2 = np.genfromtxt("content/data/DiffuserScan.UXD", unpack=True)

########################################################################

a_c = 0.223  # ° (kritischer Winkel von Polysterol)
geo_exp = 0.44  # °( aus vorheriger Berechnung)
I0 = 1.61e4 # Intensität aus vorheriger Berechnung

def r(alpha):
    return (a_c / (2 * alpha)) ** 4


# Von nun an mit Reflektivität:
R = (c1 - c2) / (5 * I0)
# Korrektur mit G-Faktor
R_c = np.array(R)
R_c[(t < geo_exp) & (t > 0)] = (
    R[(t < geo_exp) & (t > 0)]
    * np.sin(np.deg2rad(geo_exp))
    / np.sin(np.deg2rad(t[(t < geo_exp) & (t > 0)]))
)

# Berechnung der Schichtdicke aus Oszillationen
idx = []
for i in range(len(t)):
    if (
        t[i] > 0.2
        and t[i] < 0.9
        and R_c[i] <= R_c[i - 1]
        and R_c[i] < R_c[i + 1]
        and R_c[i] < R_c[i - 2]
        and R_c[i] < R_c[i + 2]
    ):
        idx.append(i)

diffs = np.diff(t[idx])
lam = 1.54e-10  # Wellenlänge der Strahlung (K_alpha Linie Kupfer)

a_d = ufloat(np.mean(diffs), np.std(diffs))
d = lam / (2 * a_d * np.pi / 180)

########################################################################
# Parratt-Algorithmus

lam = 1.54e-10
k = 2*np.pi/lam
n1 = 1
d1 = 0

# Werte ermittelt aus zuvorigem Fit
delta_Poly = 4.2e-6 # 1. Schicht Polysterol
delta_Si = 1.6e-5 # 2. Schicht Silizium
b_Poly = 2.7e-8
b_Si = 9.8e-7
d_ = 8.2e-8
sigma_Poly = 4e-10
sigma_Si = 3e-10

params = [delta_Poly, delta_Si, b_Poly, b_Si, d_, sigma_Poly, sigma_Si] # Startwerte
err = np.zeros(len(params))

def parratt(a, delta2, delta3, b2, b3, d2, sigma1, sigma2):
    n2 = 1.0 - delta2 - b2*1j
    n3 = 1.0 - delta3 - b3*1j
    a = np.deg2rad(a)
    kd1 = k *  np.sqrt(n1**2 - np.cos(a)**2)
    kd2 = k * np.sqrt(n2**2 - np.cos(a)**2)
    kd3 = k * np.sqrt(n3**2 - np.cos(a)**2)

    r12 = ((kd1 - kd2)/(kd1 + kd2))*np.exp(-2*kd1*kd2*sigma1**2)
    r23 = ((kd2 - kd3)/(kd2 + kd3))*np.exp(-2*kd2*kd3*sigma2**2)

    x2 = np.exp(-2j* kd2 * d2) * r23
    x1 = (r12 + x2)/(1+ r12*x2)

    return np.abs(x1)**2

# Fitbereich
t_min = 0.35
t_max = 0.75

bounds = ([1e-7, 1e-7, 1e-10, 1e-10, 1e-9, 5e-12, 5e-12], [5e-5, 5e-5, 1e-6, 1e-6, 1e-7, 1e-9, 1e-9]) # Limits der Parameter
#params, pcov = op.curve_fit(parratt, t[(t>t_min) * (t<t_max)], R_c[(t>t_min) * (t<t_max)], p0 = params, bounds = bounds)
#err = np.sqrt(np.diag(pcov))

delta_Si = ufloat(params[0], err[0])
delta_Poly = ufloat(params[1], err[1])
a_c_Poly = unp.sqrt(2*delta_Poly)*180/np.pi
a_c_Si = unp.sqrt(2*delta_Si)*180/np.pi
print("-------------------------------------------------------")
print("Parameter des Parrattalgorithmus")
print(f"delta_Poly  : {params[0]:.4e} +- {err[0]:.4e}")
print(f"delta_Si    : {params[1]:.4e} +- {err[1]:.4e}")
print(f"b_Poly      : {params[2]:.4e} +- {err[2]:.4e}")
print(f"b_Si        : {params[3]:.4e} +- {err[3]:.4e}")
print(f"d2          : {params[4]:.4e} +- {err[4]:.4e} m")
print(f"sigma_Poly  : {params[5]:.4e} +- {err[5]:.4e}")
print(f"sigma_Si    : {params[6]:.4e} +- {err[6]:.4e}")
print(f"alpha_c (Poly)  : {a_c_Poly:.4f} °")
print(f"alpha_c (Si)    : {a_c_Si:.4f} °")
print("-------------------------------------------------------")

x = np.linspace(0, 2.5, 1000)

plt.plot(t, R_c, label = "gemessene Reflektivität (korrigiert)", c = "cornflowerblue")
plt.plot(x, parratt(x, *params), color = "firebrick", alpha = .8, label = "Parrattalgorithmus")
#plt.vlines(noms(a_c_Poly), 0, 10e3, label = r"$\alpha_c$ (Polysterol) = " + f"{a_c_Poly:.4f}°", color = "deeppink")
#plt.vlines(noms(a_c_Si), 0, 10e3, label = r"$\alpha_c$ (Si) = " + f"{a_c_Si:.4f}°", color = "rebeccapurple")
plt.legend()
plt.yscale("log")
plt.xlim(0, 2.5)
plt.ylim(None, 10e3)
plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
plt.ylabel(r"$R$")
plt.tight_layout()
#plt.show()
plt.savefig("build/Reflek3.pdf")
plt.close()
