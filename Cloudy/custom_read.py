from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class CloudyCoolingFunc:
    """
    Class for reading and interpolating Cloudy cooling function tables

    Written by Sten Sipma
    """
    def __init__(self, data_location="./FinalTable.txt", interp_method='cubic'):
        interp_funcs, axes = CloudyCoolingFunc.create_cloudy_interp(data_location, interp_method=interp_method)

        # Unpack
        self.heating_int, self.cooling_int, self.rho_int, self.mol_weight_int = interp_funcs
        self.n_H, self.Z, self.T = axes

    def __str__(self):
        return f"CloudyCoolingFunc(limits: n_H={self.n_H[0]} to {self.n_H[-1]}, Z={self.Z[0]} to {self.Z[-1]}, T={self.T[0]} to {self.T[-1]})"

    def __call__(self, n_H, Z, T):
        """
        Given 3 1D arrays n_H, Z, T of size N, interpolate the CLOUDY
        quantities at these N points.

        Returns:
            - Heating (erg cm3 / s)
            - Cooling (erg cm3 / s)
            - Rho (g)
            - Molecular Weight
        """
        return self.heating(n_H, Z, T), self.cooling(n_H, Z, T), self.rho(n_H, Z, T), self.mol_weight(n_H, Z, T)

    # Methods for the direct interp quantities
    def heating(self, *args, **kwargs):
        return CloudyCoolingFunc._interp_3d_helper(self.heating_int, *args, **kwargs)

    def cooling(self, *args, **kwargs):
        return CloudyCoolingFunc._interp_3d_helper(self.cooling_int, *args, **kwargs)

    def rho(self, *args, **kwargs):
        return CloudyCoolingFunc._interp_3d_helper(self.rho_int, *args, **kwargs)

    def mol_weight(self, *args, **kwargs):
        return CloudyCoolingFunc._interp_3d_helper(self.mol_weight_int, *args, **kwargs)

    # Helper functions for loading
    @staticmethod
    def _interp_3d_helper(interp, n_H, Z, T):
        test_points = np.vstack([n_H, Z, T]).T
        return interp(test_points)

    @staticmethod
    def create_cloudy_interp(data_location, interp_method='cubic'):
        table_location = Path(data_location)
        data = np.loadtxt(table_location, comments="#")

        # The dimensions of each axis
        dim_T = 101
        dim_Z = 6
        dim_n_H = 7
        shape = (dim_n_H, dim_Z, dim_T)

        # All the data as a cube
        # TRANSFORM: n_h -> linear space
        #            Z   -> linear space
        n_H = 10.**data[:, 0].reshape(shape)
        Z = 10.**data[:, 1].reshape(shape)
        T = data[:, 2].reshape(shape)
        heating = data[:, 3].reshape(shape)
        cooling = data[:, 4].reshape(shape)
        mol_weight = data[:, 5].reshape(shape)
        rho = data[:, 6].reshape(shape)

        # The axes:
        n_H_ax = n_H[:, 0, 0]
        Z_ax = Z[0, :, 0]
        T_ax = T[0, 0, :]

        axes = (n_H_ax, Z_ax, T_ax)

        heating_interp = RegularGridInterpolator(axes, heating, method=interp_method)
        cooling_interp = RegularGridInterpolator(axes, cooling, method=interp_method)
        rho_interp = RegularGridInterpolator(axes, rho, method=interp_method)
        mol_weight_interp = RegularGridInterpolator(axes, mol_weight, method=interp_method)

        return (heating_interp, cooling_interp, rho_interp, mol_weight_interp), (n_H_ax, Z_ax, T_ax)

if __name__ == '__main__':
    table_location = Path("./FinalTable.txt")
    data = np.loadtxt(table_location, comments="#")

    headers = ["Hden", "Metals", "T", "Htot", "Ctot", "Molwgt", "dens"]

    # The dimensions of each axis
    dim_T = 101
    dim_Z = 6
    dim_n_H = 7
    shape = (dim_n_H, dim_Z, dim_T)

    # All the data as a cube
    n_H = data[:, 0].reshape(shape)
    Z = data[:, 1].reshape(shape)
    T = data[:, 2].reshape(shape)
    heating = data[:, 3].reshape(shape)
    cooling = data[:, 4].reshape(shape)
    mol_weight = data[:, 5].reshape(shape)
    rho = data[:, 6].reshape(shape)

    # The axes:
    n_H_ax = n_H[:, 0, 0]
    Z_ax = Z[0, :, 0]
    T_ax = T[0, 0, :]

    axes = (n_H_ax, Z_ax, T_ax)

    heating_interp = RegularGridInterpolator(axes, heating)
    cooling_interp = RegularGridInterpolator(axes, cooling)
    rho_interp = RegularGridInterpolator(axes, rho)
    mol_weight_interp = RegularGridInterpolator(axes, mol_weight)


    # print(heating_interp(test_basic), heating[i, j, k])
    # print(cooling_interp(test_basic), cooling[i, j, k])
    # print(mol_weight_interp(test_basic), mol_weight[i, j, k])


    # TEST:
    i = 5
    j = 1
    k = 0
    test_basic = [n_H_ax[i], Z_ax[j], T_ax[k]]

    cool = CloudyCoolingFunc()

    print(cool.heating(*test_basic), heating[i, j, k])
    print(cool.cooling(*test_basic), cooling[i, j, k])
    print(cool.mol_weight(*test_basic), mol_weight[i, j, k])
    print(cool.rho(*test_basic), rho[i, j, k])
