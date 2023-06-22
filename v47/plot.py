import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unp

# delta t / s, R / OHM, I / mA, U / V
delta_t, R, I_, U = np.genfromtxt("content/data/data.txt", unpack=True)

# constants
M = 63.546 #g/mol
m = 342 #g
V_0 = 7.092*1e-6#m^3/mol
B = 140*1e9 #Pa N/m^2

I = I_*1e-3
#calculating Energy
E = U*I*delta_t

#calculating T
def T(R):
    return 0.00134*R**2 + 2.296*R - 243.02

T_arr = T(R)+273.15 # T in K as array

def C_P(E,delta_T):
    return M/m * E/delta_T

C_P_values = np.zeros(len(delta_t))

for i in range(1, len(delta_t)):
    C_P_values[i] = C_P(E[i], T(R[i])-T(R[i-1]))


# converting T to alpha, interpolation
T_zualpha_array = np.array([70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310])
alpha_array = np.array([7,8.5,9.75,10.70,11.5,12.1,12.65,13.15,13.6,13.9,14.25,14.5,14.75,14.95,15.2,15.4,15.6,15.75,15.9,16.1,16.25,16.35,16.5,16.65,16.8])*10**-6
alpha = np.ones(len(delta_t))

for i in range(len(T_arr)):
    j=0
    while(T_arr[i]>T_zualpha_array[j]):
        j = j+1
    alpha[i] = (alpha_array[j]-alpha_array[j-1])*(T_arr[i]-T_zualpha_array[j-1])/(T_zualpha_array[j]-T_zualpha_array[j-1])+alpha_array[j-1]

# converting C_P to C_V
C_V_values = C_P_values - 9*T_arr*V_0*B*alpha**2

#theta_D hardcode
C_V_theta_D = C_V_values[1:9]
theta_D_T = np.array([2.4, 3.1, 2.4, 2.2, 2.4, 2.1, 2.8, 1.7])
theta_D = theta_D_T * T_arr[1:9]

theta_D_mean = np.mean(theta_D)
theta_D_std = np.std(theta_D)

print(f"mean theta_D: ({theta_D_mean:.2f} +- {theta_D_std:.2f}) K")

# save data to file
np.savetxt(
    "content/data/data_out.txt", np.array([E, T(R)+273.15, T(R), C_P_values, alpha*10**6, C_V_values]).transpose(),
     header = "E / J, T / K, T / °C, C_P / J/molK, alpha / 10^-6 1/K, C_V / J/molK",
     fmt = ["%.1f", "%d", "%d", "%.2f", "%.2f", "%.2f"],
     delimiter="\t\t"
)
np.savetxt(
    "content/data/data_out2.txt", np.array([T_arr[1:9], C_V_theta_D, theta_D_T, theta_D]).transpose(),
     header = "T / K, C_V/J/molK, theta_D/T, theta_D / K",
     fmt = ["%d", "%.2f", "%.1f", "%.2f"],
     delimiter="\t\t"
)

# plot c_p vs T
fig, ax = plt.subplots()

ax.plot(T(R[1:]), C_P_values[1:], '1', ms=12, mew=1.2, label=r"$C_\text{P}$")
ax.plot(T(R[1:]), C_V_values[1:], '1', ms=12, mew=1.2, label=r"$C_\text{V}$")

ax.set_xlabel(r"$T\,/\,\unit{\celsius}$")
ax.set_ylabel(r"$C\,/\,\unit{\joule\per\mol\per\kelvin}$")

ax.legend()
ax.grid()

fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
fig.savefig('build/C.pdf')


