import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex=True)


T = np.linspace(np.log10(1e3), np.log10(1e8), 101)
T = 10**T

print(T)

Z = np.array([-2, -1.5, -1.0, -0.5, 0.0, 0.5])

nh = np.linspace(np.log10(1e-6), np.log10(1), 7)


amu = 1.66054e-24

cooling_net = np.loadtxt("FinalTable.txt")[:, 4]
heating_net = np.loadtxt("FinalTable.txt")[:, 3]

nh_tab = np.loadtxt("FinalTable.txt")[:, 0]
rho_tab = np.loadtxt("FinalTable.txt")[:, 6]

print(10 ** (nh_tab) * amu / rho_tab)


cool_net = np.zeros((len(T), len(Z), len(nh)))
heat_net = np.zeros((len(T), len(Z), len(nh)))
nh_table = np.zeros((len(T), len(Z), len(nh)))
rho_table = np.zeros((len(T), len(Z), len(nh)))


for k in range(len(nh)):
    for j in range(len(Z)):
        for i in range(len(T)):
            l = i + j * len(T) + k * len(T) * len(Z)
            cool_net[i, j, k] = cooling_net[l]
            heat_net[i, j, k] = heating_net[l]
            rho_table[i, j, k] = rho_tab[l]
            nh_table[i, j, k] = nh_tab[l]
