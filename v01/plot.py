import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as devs
import scipy.optimize as op

# Bestimmung der Auflösungszeit

t1, c1 = np.genfromtxt("content/data/T_VZ_10s.txt", unpack = True)
t2, c2 = np.genfromtxt("content/data/T_VZ_20s.txt", unpack = True)

c1 = unp.uarray(c1, np.sqrt(c1))/10
c2 = unp.uarray(c2, np.sqrt(c2))/20

h = np.mean(np.concatenate((noms(c1[(t1 >= - 5) & (t1 <= 1)]), noms(c2[(t2 >= - 5) & (t2 <= 1)]))))

#print(x)
plt.errorbar(t1, noms(c1), yerr= devs(c1), lw = 0, elinewidth = 1, marker = ".", capsize = 1, label = "Messreihe 1", c = "firebrick")
plt.errorbar(t2, noms(c2), yerr= devs(c2), lw = 0, elinewidth = 1, marker = ".", capsize = 1, label = "Messreihe 2", c = "black")
plt.hlines(noms(h), -5, 1, colors = "cornflowerblue", label = "Plateau", lw = 1.5)
plt.hlines(noms(h)/2, -9, 5, colors = "cornflowerblue", label = "Halbwertsbreite", lw = 1.5, ls = "dashed")
plt.vlines([-9, 5], 0, noms(h)/2, colors = "cornflowerblue", lw = 1, ls = "dashed")
plt.grid()
plt.legend()
plt.xlabel(r"$T_\text{VZ} \mathbin{/} \unit{\nano\second}$")
plt.ylabel("a.u.")
plt.xlim(-15, 15)
plt.ylim(0, 5)
plt.tight_layout()
plt.show()
plt.savefig('build/plot1.pdf')
plt.close()

# Kalibrierung des VKA 

def f(x, a, b):
    return a*x + b

channel = np.array([37, 81, 126, 171, 216, 261, 306, 350, 395, 440])
time = np.array([0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8])

params, pcov = op.curve_fit(f, channel, time)
err = np.sqrt(np.diag(pcov))
#m = np.mean(time/channel)

print("Parameter VKA Kalibrierung: \n", params, err)
x = np.linspace(0, 450, 100)

plt.plot(channel, time, marker = "x", lw = 0, c = "black", ms = 8, label = "Messwerte")
plt.plot(x, f(x, *params), c = "firebrick", lw = 0.7, label = "Linearer Fit")
plt.plot([], [], " ", label = f"a = {params[0]:.2e} $\pm$ {err[0]:.2e}")
plt.plot([], [], " ", label = f"b = {params[1]:.2e} $\pm$ {err[1]:.2e}")

plt.xlabel("Kanalnummer")
plt.ylabel(r'$t \mathbin{/} \unit{\micro\second}$')
plt.legend()
plt.grid()
plt.xlim(0, 450)
plt.ylim(0, 10)
plt.tight_layout()
#plt.show()
plt.savefig('build/plot2.pdf')
plt.close()

# Bestimmung der Lebensdauer

counts = np.genfromtxt("content/data/messwerte.txt")
channel = np.array(range(len(counts)))
t = f(channel, *unp.uarray(params, err))

# Fit
def model(t, N_0, tau, U_0):
    return N_0* np.exp(-t/tau) + U_0

mask = [(counts > 0) & (counts < 500)] # Ausreißer
params, pcov = op.curve_fit(model, noms(t[mask]), counts[mask])
err = np.sqrt(np.diag(pcov))

print("Parameter des Fits: \n", f"N_0:  {params[0]} +- {err[0]} \n tau:  {params[1]} +- {err[1]} \n U_0:  {params[2]} +- {err[2]}")

# Plotting

x = np.linspace(0, 13, 1000)

plt.errorbar(noms(t), counts, xerr=devs(t), yerr=np.sqrt(counts), lw=0, elinewidth=1, capsize=2, c = "black", label = "Messdaten", alpha = .6)
plt.plot(x, model(x, *params), label = "Fit", c = "firebrick")

plt.yscale("log")
plt.ylim(1, 1e3)
plt.xlim(0, 11)
plt.ylabel(r"log($N$)")
plt.xlabel(r'$t \mathbin{/} \unit{\micro\second}$')
plt.legend()
plt.grid()
#plt.show()
plt.tight_layout()
plt.savefig("build/fit_log.pdf")
plt.close()

plt.errorbar(noms(t), counts, xerr=devs(t), yerr=np.sqrt(counts), lw=0, elinewidth=1, capsize=2, c = "black", label = "Messdaten", alpha = .6)
plt.plot(x, model(x, *params), label = "Fit", c = "firebrick")

plt.ylim(0, 540)
plt.xlim(0, 11)
plt.ylabel(r"$N$")
plt.xlabel(r'$t \mathbin{/} \unit{\micro\second}$')
plt.legend()
plt.grid()
#plt.show()
plt.tight_layout()
plt.savefig("build/fit.pdf")
plt.close()

#x = np.linspace(0, 10, 1000)
#y = x ** np.sin(x)
#
#plt.subplot(1, 2, 1)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')
#
#plt.subplot(1, 2, 2)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')
#
## in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/plot.pdf')
