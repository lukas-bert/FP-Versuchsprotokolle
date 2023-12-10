import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const

# Plot for the stability paramter for different configurations 

r1 = np.inf
r2 = 1 # meter
r3 = 1.4 # meter
  
def g(L, r):
    return 1 - L/r

fig1, ax = plt.subplots()

l = np.linspace(0, 2.5, 10000)

ax.hlines(0, 0, 2.5, ls = "dashed", colors = "grey")
ax.plot(l, g(l, r1)*g(l, r1), label = "flat / flat")
ax.plot(l, g(l, r2)*g(l, r1), label = "r = 1000 mm / flat")
ax.plot(l, g(l, r3)*g(l, r1), label = "r = 1400 mm / flat")
ax.plot(l, g(l, r2)*g(l, r2), label = "r = 1000 mm / r = 1000 mm", ls = "dotted")
ax.plot(l, g(l, r3)*g(l, r3), label = "r = 1400 mm / r = 1400 mm", ls = "dotted")

ax.plot(1, 0, marker = "x", lw = 0, c = "black", ms = 10)
ax.plot(1.4, 0, marker = "x", lw = 0, c = "black", ms = 10)

ax.legend()
ax.set_ylabel(r"$g_1 g_2$")
ax.set_xlabel(r"$L$ / m")
ax.set_xlim(0, 2.5)
ax.set_ylim(-1.5, 2.5)
plt.grid()
plt.tight_layout()

plt.show()

fig1.savefig("build/stability.pdf")
