"""Poisson solver module."""

from grid.interpolate import compute_spline_point_value, generate_real_sph_harms
from grid.ode import ODE

import numpy as np

from scipy.interpolate import CubicSpline


class Poisson:
    """Poisson ODE solver class."""

    @staticmethod
    def _proj_sph_value(radial, coors, l_max, value_array, weights, indices):
        """Compute the spline for target function on each spherical harmonics.

        Parameters
        ----------
        radial : RadialGrid
            Radial grids for compute coeffs on each Real Spherical Harmonics.
        coors : numpy.ndarray(N, 2)
            Spherical coordinates for comput coeff. [azimuthal, polar]
        l_max : int, >= 0
            The maximum value l in generated real spherical harmonics
        value_array : np.ndarray(N)
            Function value need to be projected onto real spherical harmonics
        weights : np.ndarray
            Weight of each point on the integration grid
        indices : np.ndarray
            Array of indices indicate the beginning and ending of each
            radial point

        Returns
        -------
        np.ndarray[scipy.PPoly], shape(2L - 1, L + 1)
            scipy cubic spline instance of each harmonic shell
        """
        if coors.shape[1] > 2:
            raise ValueError(
                f"Input coordinates contains too many columns\n"
                f"Only 2 columns needed, got coors shape:{coors.shape}"
            )
        theta, phi = coors[:, 0], coors[:, 1]
        real_sph = generate_real_sph_harms(l_max, theta, phi)
        # real_sph shape: (m, l, n)
        ms, ls = real_sph.shape[:-1]
        # store spline for each p^{lm}
        spls_mtr = np.zeros((ms, ls), dtype=object)
        ml_sph_value = compute_spline_point_value(
            real_sph, value_array, weights, indices
        )
        ml_sph_value /= (radial.points ** 2 * radial.weights)[:, None, None]
        for l_value in range(ls):
            for m_value in range(-l_value, l_value + 1):
                spls_mtr[m_value, l_value] = CubicSpline(
                    x=radial.points, y=ml_sph_value[:, m_value, l_value]
                )
        return spls_mtr

    @staticmethod
    def solve_poisson(spls_mtr, x_range, boundary, tfm=None):
        """Solve poisson equation for each spherical harmonics.

        Parameters
        ----------
        spls_mtr : np.ndarray(M, L), M = 2l + 1, L = l + 1
            An array of callable or spline of each shell of harmonics
        x_range : np.ndarray(K,)
            An array with points for numerically solve the ODE
            The boundary point should be within radial points range
        boundary : float
            Boundary value when x to go infinite
        tfm : None, optional
            Transformation for given x variable

        Returns
        -------
        np.ndarray(M, L)
            Solved spline for each spherical harnomics
        """
        # store V^{lm}
        ms, ls = spls_mtr.shape
        res_mtr = np.zeros((ms, ls), dtype=object)  # result_matrix
        for l_v in range(ls):  # l_v: value of L
            for m_v in range(-l_v, l_v + 1):  # m_v: value of M
                res = Poisson.solve_poisson_bv(
                    spls_mtr[m_v, l_v], x_range, boundary, (m_v, l_v), tfm=tfm
                )
                res_mtr[m_v, l_v] = res
        return res_mtr

    @staticmethod
    def solve_poisson_bv(fx, x_range, boundary, m_l=(0, 0), tfm=None):
        """Solve poisson equation for given function.

        .. math::

        Parameters
        ----------
        fx : Callable
            Callable function on the right hand side of equation
        x_range : np.narray(K,)
            An array with points for numerically solve the ODE
            The boundary point should be within radial points range
        boundary : float
            Boundary value when x to go infinite
        m_l : tuple, optional
            m and l value of given spherical harmonics
        tfm : None, optional
            Transformation for given x variable

        Returns
        -------
        scipy.PPline
            A callable spline for result.
        """
        m_value, l_value = m_l

        def f_x(r):
            return fx(r) * -4 * np.pi * r

        def coeff_0(r):
            return -l_value * (l_value + 1) / r ** 2

        coeffs = [coeff_0, 0, 1]
        if l_value == 0 and m_value == 0:
            bd_cond = [(0, 0, 0), (1, 0, boundary)]
        else:
            bd_cond = [(0, 0, 0), (1, 0, 0)]
        return ODE.solve_ode(x_range, f_x, coeffs, bd_cond, transform=tfm)

    @staticmethod
    def interpolate_radial(spls_mtr, rad, deriv=0, sumup=False):
        """Interpolate radial value for given set of splines.

        Parameters
        ----------
        spls_mtr : np.ndadrray(M, L)
            An array of solved spline for poisson ODE
        rad : float
            Radial value to be interpolated
        deriv : int, optional
            0 for function value, 1 for its first order deriv
        sumup : bool, optional
            False: return an array(M, L) contains the coeff for each shell
            True: sum the coeff of each shell
        Returns
        -------
        np.ndarray(M, L) or float
            an array of coeffs of each shell or the sum of all the coeffs
        """
        ms, ls = spls_mtr.shape
        inter_mtr = np.zeros((ms, ls))
        for l_v in range(ls):
            for m_v in range(-l_v, l_v + 1):
                inter_mtr[m_v, l_v] = spls_mtr[m_v, l_v](rad)[deriv]
        if sumup:
            return np.sum(inter_mtr)
        else:
            return inter_mtr

    @staticmethod
    def interpolate(spls_mtr, rad, theta, phi, deriv=0):
        """Inperpolate points on any 3d space.

        Parameters
        ----------
        spls_mtr : np.ndadrray(M, L)
            An array of solved spline for poisson ODE
        rad : float
            Radial value to be interpolated
        theta : np.ndarray(N,)
            An array of azimuthal angles
        phi : np.ndarray(N,)
            An array of polar angles
        deriv : int, optional
            0 for function value, 1 for its first order deriv

        Returns
        -------
        np.ndarray(N,)
            Interpolated value on each point
        """
        ms, ls = spls_mtr.shape
        # (M, L, N)
        sph_harm = generate_real_sph_harms(ls - 1, theta, phi)
        r_value = Poisson.interpolate_radial(spls_mtr, rad, deriv)
        return np.sum(r_value[..., None] * sph_harm, axis=(0, 1))

    # Doesn't give the right answer. Commented out for now.
    # @staticmethod
    # def solve_poisson_iv(fx, x_range, m_l=(0, 0), tfm=None):
    #     m_value, l_value = m_l

    #     def f_x(r):
    #         return -4 * np.pi * fx(r)

    #     def coeff_1(r):
    #         return 2 / r

    #     def coeff_0(r):
    #         return -l_value * (l_value + 1) / r ** 2

    #     coeffs = [coeff_0, coeff_1, 1]
    #     bd_cond = [(1, 0, 0), (1, 1, 0)]
    #     return ODE.solve_ode(x_range, f_x, coeffs, bd_cond, transform=tfm)
