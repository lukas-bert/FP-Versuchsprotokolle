import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const

# Plot zum B-Feld

x, B = np.genfromtxt("content/data/b_field.txt", unpack = True)

plt.plot(x, B, marker = "x", color = "firebrick", label = "Messwerte", lw = 0)
plt.plot(x, B, color = "gray",lw = 0.5, ls = "dashed")
plt.plot([], [], " ", label = r"Maximum: \qty{412}{\milli\tesla}")
plt.grid()
plt.xlabel(r"$x \mathbin{/} \unit{\milli\metre}$")
plt.ylabel(r"$B \mathbin{/} \unit{\milli\tesla}$")
plt.ylim(0, 440)
plt.xlim(70, 135)
plt.legend()
plt.tight_layout()
#plt.show()

plt.savefig('build/magnetfeld.pdf')
plt.close()

y, t1, m1, t2, m2 = np.genfromtxt("content/data/GaAs12.txt", unpack = True)
# umrechnen der Minuten

t1_12 = (t1 + m1/60)
t2_12 = (t2 + m2/60)
theta12 = (t1_12 - t2_12)/2

y, t1, m1, t2, m2 = np.genfromtxt("content/data/GaAs28.txt", unpack = True)
t1_28 = (t1 + m1/60)
t2_28 = (t2 + m2/60)
theta28 = (t1_28 - t2_28)/2

y, t1, m1, t2, m2 = np.genfromtxt("content/data/GaAs_pure.txt", unpack = True)
t1_p = (t1 + m1/60)
t2_p = (t2 + m2/60)
theta_p = (t1_p - t2_p)/2

np.savetxt(
    "content/data/differenzen.txt", np.array([theta12, t1_12, t2_12,  theta28, t1_28, t2_28, theta_p, t1_p, t2_p]).transpose(),
     header = "GaAs12 (theta, theta1, theta2),   GaAs28 (theta, theta1, theta2),    GaAs (rein) (theta, theta1, theta2)", fmt = "%.2f"
)

# Umrechnen in Bogenmaß
l12 = 1.36e-3
l28 = 1.296e-3
l_p = 5.11e-3

theta12 = (theta12/180 * np.pi)/l12
theta28 = (theta28/180 * np.pi)/l28
theta_p = (theta_p/180 * np.pi)/l_p

plt.plot(y**2, theta12, lw = 0, c = "indigo", marker = "x", label = r"GaAs n-dotiert, N = $\qty{1.2e18}{\per\cubic\centi\metre}$", ms = 6)
plt.plot(y**2, theta28, lw = 0, c = "peru", marker = "1", label = r"GaAs n-dotiert, N = $\qty{2.8e18}{\per\cubic\centi\metre}$", ms = 9)
plt.plot(y**2, theta_p, lw = 0, c = "forestgreen", marker = "+", label = r"GaAs (undotiert)$", ms = 8)
plt.grid()
plt.xlim(1, 7.5)
plt.ylabel(r"$\theta \mathbin{/} d \mathbin{/} \unit{\radian\per\metre}$")
plt.xlabel(r"$\lambda^2 \mathbin{/} \unit{\micro\metre\squared}$")
plt.legend()
plt.tight_layout()

#plt.show()
plt.savefig('build/messwerte.pdf')
plt.close()


# Bestimmung der effektiven Masse

def f(x, a, b):
    return (a*x + b) 

params1, pcov1 = op.curve_fit(f, y**2, theta12-theta_p)
err1 = np.sqrt(np.diag(pcov1))
a1 = ufloat(params1[0], err1[0])*10**(12)
b1 = ufloat(params1[1], err1[1])

params2, pcov2 = op.curve_fit(f, y**2,theta28-theta_p)
err2 = np.sqrt(np.diag(pcov2))
a2 = ufloat(params2[0], err2[0])*10**(12)
b2 = ufloat(params2[1], err2[1])

print("------------------------------------------------------")
print(f"a12 = {a1:.3e} m^-3")
print(f"b12 = {b1:.3e} m^-1", "\n")
print(f"a28 = {a2:.3e} m^-3")
print(f"b28 = {b2:.3e} m^-1")
print("------------------------------------------------------")

# Ladungsträgerdichten pro m^3
N12 = 1.2e18*10**6
N28 = 2.8e18*10**6
n = 3.354
B = 412e-3

m1 = unp.sqrt(const.e**3 * N12*B/(8*const.pi**2* const.epsilon_0*const.c**3*n*a1))
m2 = unp.sqrt(const.e**3 * N28*B/(8*const.pi**2* const.epsilon_0*const.c**3*n*a2))
m = (m1 + m2)/2

m_lit = 0.063*const.m_e

# relative Abweichungen
delta1 = (m1 -m_lit)/m_lit*100
delta2 = (m2 -m_lit)/m_lit*100
delta3 = (m -m_lit)/m_lit*100

print("------------------------------------------------------")
print(f"m12 = {m1:.4e} kg           delta = {delta1:.3f}")
print(f"m28 = {m2:.4e} kg           delta = {delta2:.3f}")
print(f"Mittelwert = {m:.4e} kg    delta = {delta3:.3f}")
print(f"Literaturwert = {m_lit:.4e} kg")
print("------------------------------------------------------")

# Plot

x = np.linspace(0, 8, 1000)

plt.plot(x, f(x, *params1), label = "Regression", ls = "dashed")
plt.plot(x, f(x, *params2), label = "Regression", ls = "dashed")

plt.plot(y**2, theta12-theta_p, lw = 0, c = "indigo", marker = "x", label = r"GaAs n-dotiert, N = $\qty{1.2e18}{\per\cubic\centi\metre}$", ms = 6)
plt.plot(y**2, theta28-theta_p, lw = 0, c = "peru", marker = "1", label = r"GaAs n-dotiert, N = $\qty{2.8e18}{\per\cubic\centi\metre}$", ms = 9)

plt.grid()
plt.ylim(0, 115)
plt.xlim(1, 7.5)
plt.ylabel(r"$\theta_\text{frei} \mathbin{/} \unit{\radian\per\metre}$")
plt.xlabel(r"$\lambda^2 \mathbin{/} \unit{\micro\metre\squared}$")
plt.legend()
plt.tight_layout()

#plt.show()
plt.savefig('build/fit.pdf')
plt.close()
