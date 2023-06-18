import matplotlib.pyplot as plt
import numpy as np

# delta t / s, R / OHM, I / mA, U / V
delta_t, R, I_, U = np.genfromtxt("content/data/data.txt", unpack=True)

I = I_*1e-3
#calculating Energy
E = U*I*delta_t

#calculating T
def T(R):
    return 0.00134*R**2 + 2.296*R - 243.02

np.savetxt(
    "content/data/E.txt", np.array([E, T(R)+273.15, T(R)]).transpose(),
     header = "E / J, T / K, T / °C",
     fmt = ["%.1f", "%d", "%d"],
     delimiter="\t\t"
)

# in matplotlibrc leider (noch) nicht möglich
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/plot.pdf')
