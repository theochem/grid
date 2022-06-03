"""Poisson solver module."""

from grid.atomgrid import AtomGrid
from grid.utils import generate_real_spherical_harmonics
from grid.ode import solve_ode

import numpy as np

from scipy.interpolate import CubicSpline


class Poisson:
    """Poisson ODE solver class."""

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
        return solve_ode(x_range, f_x, coeffs, bd_cond, transform=tfm)

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


def interpolate_laplacian(atomgrid: AtomGrid, func_vals: np.ndarray):
    r"""
    Return a function that interpolates the Laplacian of a function.

    .. math::
        \Deltaf = \frac{1}{r}\frac{\partial^2 rf}{\partial r^2} - \frac{\hat{L}}{r^2},

    such that the angular momentum operator satisfies :math:`\hat{L}(Y_l^m) = l (l + 1) Y_l^m`.
    Expanding f in terms of spherical harmonic expansion, we get that

    .. math::
        \Deltaf = \sum \sum \bigg[\frac{\rho_{lm}^{rf}}{r} -
        \frac{l(l+1) \rho_{lm}^f}{r^2} \bigg] Y_l^m,

    where :math:`\rho_{lm}^f` is the lth, mth radial component of function f.

    Parameters
    ----------
    func_vals : ndarray(N,)
        The function values evaluated on all :math:`N` points on the atomic grid.

    Returns
    -------
    callable[ndarray(M,3) -> ndarray(M,)] :
        Function that interpolates the Laplacian of a function whose input is
        Cartesian points.

    """
    radial, theta, phi = atomgrid.convert_cartesian_to_spherical().T
    # Multiply f by r to get rf
    func_vals_radial = radial * func_vals

    # compute spline for the radial components for rf and f
    radial_comps_rf = atomgrid.radial_component_splines(func_vals_radial)
    radial_comps_f = atomgrid.radial_component_splines(func_vals)

    def interpolate_laplacian(points):
        r_pts, theta, phi = atomgrid.convert_cartesian_to_spherical(points).T

        # Evaluate the radial components for function f
        r_values_f = np.array([spline(r_pts) for spline in radial_comps_f])
        # Evaluate the second derivatives of splines of radial component of function rf
        r_values_rf = np.array([spline(r_pts, 2) for spline in radial_comps_rf])

        r_sph_harm = generate_real_spherical_harmonics(atomgrid.l_max // 2, theta, phi)

        # Divide \rho_{lm}^{rf} by r  and \rho_{lm}^{rf} by r^2
        # l means the angular (l,m) variables and n represents points.
        with np.errstate(divide='ignore', invalid='ignore'):
            r_values_rf /= r_pts
            r_values_rf[:, r_pts < 1e-10] = 0.0
            r_values_f /= r_pts**2.0
            r_values_f[:, r_pts**2.0 < 1e-10] = 0.0

        # Multiply \rho_{lm}^f by l(l+1), note 2l+1 orders for each degree l
        degrees = np.hstack([[x * (x + 1)] * (2 * x + 1) for x in
                             np.arange(0, atomgrid.l_max // 2 + 1)])
        second_component = np.einsum("ln,l->ln", r_values_f, degrees)
        # Compute \rho_{lm}^{rf}/r - \rho_{lm}^f l(l + 1) / r^2
        component = r_values_rf - second_component
        # Multiply by spherical harmonics and take the sum
        return np.einsum("ln, ln -> n", component, r_sph_harm)

    return interpolate_laplacian