# GRID is a numerical integration module for quantum chemistry.
#
# Copyright (C) 2011-2019 The GRID Development Team
#
# This file is part of GRID.
#
# GRID is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# GRID is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""Utils function module."""
import numpy as np

from scipy.special import sph_harm


_bragg = np.array(
    [
        np.nan,  # index 0, place holder
        0.472_431_53,
        np.nan,
        2.740_102_89,
        1.984_212_44,
        1.606_267_21,
        1.322_808_29,
        1.228_321_99,
        1.133_835_68,
        0.944_863_07,
        np.nan,
        3.401_507_04,
        2.834_589_2,
        2.362_157_67,
        2.078_698_75,
        1.889_726_13,
        1.889_726_13,
        1.889_726_13,
        np.nan,
        4.157_397_49,
        3.401_507_04,
        3.023_561_81,
        2.645_616_59,
        2.551_130_28,
        2.645_616_59,
        2.645_616_59,
        2.645_616_59,
        2.551_130_28,
        2.551_130_28,
        2.551_130_28,
        2.551_130_28,
        2.456_643_97,
        2.362_157_67,
        2.173_185_05,
        2.173_185_05,
        2.173_185_05,
        np.nan,
        4.440_856_41,
        3.779_452_27,
        3.401_507_04,
        2.929_075_51,
        2.740_102_89,
        2.740_102_89,
        2.551_130_28,
        2.456_643_97,
        2.551_130_28,
        2.645_616_59,
        3.023_561_81,
        2.929_075_51,
        2.929_075_51,
        2.740_102_89,
        2.740_102_89,
        2.645_616_59,
        2.645_616_59,
        np.nan,
        4.913_287_95,
        4.062_911_19,
        3.684_965_96,
        3.495_993_35,
        3.495_993_35,
        3.495_993_35,
        3.495_993_35,
        3.495_993_35,
        3.495_993_35,
        3.401_507_04,
        3.307_020_73,
        3.307_020_73,
        3.307_020_73,
        3.307_020_73,
        3.307_020_73,
        3.307_020_73,
        3.307_020_73,
        2.929_075_51,
        2.740_102_89,
        2.551_130_28,
        2.551_130_28,
        2.456_643_97,
        2.551_130_28,
        2.551_130_28,
        2.551_130_28,
        2.834_589_2,
        3.590_479_65,
        3.401_507_04,
        3.023_561_81,
        3.590_479_65,
        np.nan,
        np.nan,
    ]
)


_cambridge = np.array(
    [
        np.nan,  # index 0, place holder
        0.585_815_1,
        0.529_123_32,
        2.418_849_45,
        1.814_137_09,
        1.587_369_95,
        1.436_191_86,
        1.341_705_56,
        1.247_219_25,
        1.077_143_9,
        1.096_041_16,
        3.136_945_38,
        2.664_513_85,
        2.286_568_62,
        2.097_596_01,
        2.022_006_96,
        1.984_212_44,
        1.927_520_66,
        2.003_109_7,
        3.836_144_05,
        3.325_918,
        3.212_534_43,
        3.023_561_81,
        2.891_280_98,
        2.626_719_33,
        2.626_719_33,
        2.494_438_5,
        2.381_054_93,
        2.343_260_41,
        2.494_438_5,
        2.305_465_88,
        2.305_465_88,
        2.267_671_36,
        2.248_774_1,
        2.267_671_36,
        2.267_671_36,
        2.192_082_32,
        4.157_397_49,
        3.684_965_96,
        3.590_479_65,
        3.307_020_73,
        3.099_150_86,
        2.910_178_25,
        2.777_897_42,
        2.759_000_16,
        2.683_411_11,
        2.626_719_33,
        2.740_102_89,
        2.721_205_63,
        2.683_411_11,
        2.626_719_33,
        2.626_719_33,
        2.607_822_06,
        2.626_719_33,
        2.645_616_59,
        4.610_931_77,
        4.062_911_19,
        3.911_733_1,
        3.855_041_31,
        3.836_144_05,
        3.798_349_53,
        3.760_555_01,
        3.741_657_75,
        3.741_657_75,
        3.703_863_22,
        3.666_068_7,
        3.628_274_18,
        3.628_274_18,
        3.571_582_39,
        3.590_479_65,
        3.533_787_87,
        3.533_787_87,
        3.307_020_73,
        3.212_534_43,
        3.061_356_34,
        2.853_486_46,
        2.721_205_63,
        2.664_513_85,
        2.570_027_54,
        2.570_027_54,
        2.494_438_5,
        2.740_102_89,
        2.759_000_16,
        2.796_794_68,
        2.645_616_59,
        2.834_589_2,
        2.834_589_2,
    ]
)

_alvarez = np.array(
    [
        np.nan,  # index 0, place holder
        0.5858151,
        0.52912332,
        2.41884945,
        1.81413709,
        1.58736995,
        1.43619186,
        1.34170556,
        1.24721925,
        1.0771439,
        1.09604116,
        3.13694538,
        2.66451385,
        2.28656862,
        2.09759601,
        2.02200696,
        1.98421244,
        1.92752066,
        2.0031097,
        3.83614405,
        3.325918,
        3.21253443,
        3.02356181,
        2.89128098,
        2.62671933,
        2.62671933,
        2.4944385 ,
        2.38105493,
        2.34326041,
        2.4944385,
        2.30546588,
        2.30546588,
        2.26767136,
        2.2487741 ,
        2.26767136,
        2.26767136,
        2.19208232,
        4.15739749,
        3.68496596,
        3.59047965,
        3.30702073,
        3.09915086,
        2.91017825,
        2.77789742,
        2.75900016,
        2.68341111,
        2.62671933,
        2.74010289,
        2.72120563,
        2.68341111,
        2.62671933,
        2.62671933,
        2.60782206,
        2.62671933,
        2.64561659,
        4.61093177,
        4.06291119,
        3.9117331 ,
        3.85504131,
        3.83614405,
        3.79834953,
        3.76055501,
        3.74165775,
        3.74165775,
        3.70386322,
        3.6660687 ,
        3.62827418,
        3.62827418,
        3.57158239,
        3.59047965,
        3.53378787,
        3.53378787,
        3.30702073,
        3.21253443,
        3.06135634,
        2.85348646,
        2.72120563,
        2.66451385,
        2.57002754,
        2.57002754,
        2.4944385 ,
        2.74010289,
        2.75900016,
        2.79679468,
        2.64561659,
        2.8345892 ,
        2.8345892 ,
        4.91328795,
        4.17629476,
        4.06291119,
        3.89283584,
        3.77945227,
        3.70386322,
        3.59047965,
        3.53378787,
        3.40150704,
        3.19363717
    ]
) 
       
        



def get_cov_radii(atnums, type="bragg"):
    """Get the covalent radii for given atomic number(s).

    Parameters
    ----------
    atnums : int or np.ndarray
        atomic number of interested
    type : str, default to bragg
        types of covalent radii for elements.
        "bragg": Bragg-Slater empirically measured covalent radii
        "cambridge": Covalent radii from analysis of the Cambridge Database"
        "alvarez": Covalent radii from https://doi.org/10.1039/B801115J

    Returns
    -------
    np.ndarray
        covalent radii of desired atom(s)

    Raises
    ------
    ValueError
        Invalid covalent type, or input atomic number is 0
    """
    if isinstance(atnums, (int, np.integer)):
        atnums = np.array([atnums])
    if np.any(np.array(atnums) == 0):
        raise ValueError("0 is not a valid atomic number")
    if type == "bragg":
        return _bragg[atnums]
    elif type == "cambridge":
        return _cambridge[atnums]
    elif type == "alvarez":
        return _alvarez[atnums]
    else:
        raise ValueError(f"Not supported radii type, got {type}")


def generate_real_spherical_harmonics_scipy(
    l_max: int, theta: np.ndarray, phi: np.ndarray
):
    r"""Generate real spherical harmonics up to degree :math:`l` and for all orders :math:`m`.

    The real spherical harmonics are defined as a function
    :math:`Y_{lm} : L^2(S^2) \rightarrow \mathbb{R}` such that

    .. math::
        Y_{lm} = \begin{cases}
            \frac{i}{\sqrt{2}} (Y^m_l - (-1)^m Y_l^{-m} & \text{if } < m < 0 \\
            Y_l^0 & \text{if } m = 0 \\
            \frac{1}{\sqrt{2}} (Y^{-|m|}_{l} + (-1)^m Y_l^m) & \text{if } m > 0
        \end{cases},

    where :math:`l \in \mathbb{N}`, :math:`m \in \{-l_{max}, \cdots, l_{max} \}` and
    :math:`Y^m_l` is the complex spherical harmonic.

    Parameters
    ----------
    l_max : int
        Largest angular degree of the spherical harmonics.
    theta : np.ndarray(N,)
        Azimuthal angle :math:`\theta \in [0, 2\pi]` that are being evaluated on.
        If this angle is outside of bounds, then periodicity is used.
    phi : np.ndarray(N,)
        Polar angle :math:`\phi \in [0, \pi]` that are being evaluated on.
        If this angle is outside of bounds, then periodicity is used.

    Returns
    -------
    ndarray((l_max + 1)**2, N)
        Value of real spherical harmonics of all orders :math:`m`,and degree
        :math:`l` spherical harmonics. For each degree, the zeroth order
        is stored, followed by positive orders then negative orders,ordered as:
        :math:`(-l, -l + 1, \cdots -1)`.

    Examples
    --------
    To obtain the l-th degree for all orders
    >>> spherical_harmonic = generate_real_spherical_harmonics(5, theta, phi)
    To obtain specific degrees, e.g. l=2
    >>> desired_degree = 2
    >>> spherical_harmonic[(desired_degree)**2: (desired_degree + 1)**2, :]

    """
    if l_max < 0:
        raise ValueError(f"lmax needs to be >=0, got l_amx={l_max}")

    total_sph = np.zeros((0, len(theta)), dtype=float)
    l_list = np.arange(l_max + 1)
    for l_val in l_list:
        # generate m=0 real sphericla harmonic
        zero_real_sph = sph_harm(0, l_val, theta, phi).real

        # generate order m=positive real spherical harmonic
        m_list_p = np.arange(1, l_val + 1, dtype=float)
        pos_real_sph = (
            sph_harm(m_list_p[:, None], l_val, theta, phi).real
            * np.sqrt(2)
            * (-1) ** m_list_p[:, None]  # Remove Conway phase from SciPy
        )
        # generate order m=negative real spherical harmonic
        m_list_n = np.arange(-1, -l_val - 1, -1, dtype=float)
        neg_real_sph = (
            sph_harm(m_list_p[:, None], l_val, theta, phi).imag
            * np.sqrt(2)
            * (-1) ** m_list_n[:, None]  # Remove Conway phase from SciPy
        )

        # Convert to horton 2 order
        horton_ord = [[pos_real_sph[i], neg_real_sph[i]] for i in range(0, l_val)]
        horton_ord = tuple(x for sublist in horton_ord for x in sublist)
        total_sph = np.vstack((total_sph, zero_real_sph) + horton_ord)
    return total_sph


def generate_real_spherical_harmonics(l_max: int, theta: np.ndarray, phi: np.ndarray):
    r"""
    Compute the real spherical harmonics recursively up to a maximum angular degree l.

    .. math::
        Y_l^m(\theta, \phi) = \frac{(2l + 1) (l - m)!}{4\pi (l + m)!} f(m, \theta)
        P_l^m(\cos(\phi)),

    where :math:`l` is the angular degree, :math:`m` is the order and
    :math:`f(m, \theta) = \sqrt{2} \cos(m \theta)` when :math:`m>0` otherwise
    :math:`f(m, \theta) = \sqrt{2} \sin(m\theta)`
    when :math:`m<0`, and equal to one when :math:`m= 0`.  :math:`P_l^m` is the associated
    Legendre polynomial without the Conway phase factor.
    The angle :math:`\theta \in [0, 2\pi]` is the azimuthal angle and :math:`\phi \in [0, \pi]`
    is the polar angle.

    Parameters
    ----------
    l_max : int
        Largest angular degree of the spherical harmonics.
    theta : np.ndarray(N,)
        Azimuthal angle :math:`\theta \in [0, 2\pi]` that are being evaluated on.
        If this angle is outside of bounds, then periodicity is used.
    phi : np.ndarray(N,)
        Polar angle :math:`\phi \in [0, \pi]` that are being evaluated on.
        If this angle is outside of bounds, then periodicity is used.

    Returns
    -------
    ndarray((l_max + 1)**2, N)
        Value of real spherical harmonics of all orders :math:`m`,and degree
        :math:`l` spherical harmonics. For each degree :math:`l`, the orders :math:`m` are
        in Horton 2 order, i.e. :math:`m=0, 1, -1, 2, -2, \cdots, l, -l`.

    Notes
    -----
    - The associated Legendre polynomials are computed using the forward recursion:
      :math:`P_l^m(\phi) = \frac{2l + 1}{l - m + 1}\cos(\phi) P_{l-1}^m(x) -
      \frac{(l + m)}{l - m + 1} P_{l-1}^m(x)`, and
      :math:`P_l^l(\phi) = (2l + 1) \sin(\phi) P_{l-1}^{l-1}(\phi)`.
    - For higher maximum degree :math:`l_{max} > 1900` with double precision the computation
      of spherical harmonic will underflow within the range
      :math:`20^\circ \leq \phi \leq 160^\circ`.  This code does not implement the
      modified recursion formulas in [2] and instead opts for higher precision defined
      by the computer system and np.longdouble.
    - The mapping from :math:`(l, m)` to the correct row index in the spherical harmonic array is
      given by :math:`(l + 1)^2 + 2 * m - 1` if :math:`m > 0`, else it is :math:`(l+1)^2 + 2 |m|`.

    References
    ----------
    .. [1] Colombo, Oscar L. Numerical methods for harmonic analysis on the sphere.
       Ohio State Univ Columbus Dept of Geodetic Science And Surveying, 1981.
    .. [2] Holmes, Simon A., and Will E. Featherstone. "A unified approach to the Clenshaw
       summation and the recursive computation of very high degree and order normalised
       associated Legendre functions." Journal of Geodesy 76.5 (2002): 279-299.

    """
    numb_pts = len(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    spherical_harm = np.zeros(((l_max + 1) ** 2, numb_pts))

    # Forward recursion requires P_{l-1}^m, P_{l-2}^m, these are the two columns, respectively
    # the rows are the order m which ranges from 0 to l_max and p_leg[:l, :] gets updated every l
    p_leg = np.zeros((l_max + 1, 2, numb_pts), dtype=np.longdouble)
    p_leg[0, :, :] = 1.0  # Initial conditions: P_0^0 = 1.0

    # the coefficients of the forward recursions and initial factor of spherical harmonic.
    a_k = lambda l, m: (2.0 * (l - 1.0) + 1) / ((l - 1.0) - m + 1.0)
    b_k = lambda l, m: (l - 1.0 + m) / (l - m)
    fac_sph = lambda l, m: np.sqrt(
        (2.0 * l + 1) / (4.0 * np.pi)
    )  # Note (l-m)!/(l+m)! is moved

    # Go through each degree and then order and fill out
    spherical_harm[0, :] = fac_sph(0, 0)  # Compute Y_0^0
    i_sph = 1  # Index to start of spherical_harm
    for l_deg in range(1, l_max + 1):
        for m_ord in range(0, l_deg + 1):
            if l_deg == m_ord:
                # Do diagonal spherical harmonic Y_l^m, when l=m.
                # Diagonal recursion: P_m^m = sin(phi) * P_{m-1}^{m-1} * (2 (l - 1) + 1)
                p_leg[m_ord, 0] = (
                    p_leg[m_ord - 1, 1] * (2 * (l_deg - 1.0) + 1) * sin_phi
                )
            else:
                # Do forward recursion here and fill out Y_l^m and Y_l^{-m}
                # Compute b_k P_{l-2}^m,  since m < l, then m < l - 2
                second_fac = (
                    b_k(l_deg, m_ord) * p_leg[m_ord, 1] if m_ord <= l_deg - 2 else 0
                )
                # Update/re-define P_{l-2}^m to be equal to P_{l-1}^m
                p_leg[m_ord, 1, :] = p_leg[m_ord, 0]
                # Then update P_{l-1}^m := P_l^m := a_k cos(\phi) P_{l-1}^m - b_k P_{l, -2}^m
                p_leg[m_ord, 0, :] = (
                    a_k(l_deg, m_ord) * cos_phi * p_leg[m_ord, 0] - second_fac
                )
            # Compute Y_l^{m} that has cosine(theta) and Y_l^{-m} that has sin(theta)
            if m_ord == 0:
                # init factorial needed to compute (l-m)!/(l+m)!
                factorial = (l_deg + 1.0) * l_deg
                spherical_harm[i_sph, :] = fac_sph(l_deg, m_ord) * p_leg[m_ord, 0]
            else:
                common_fact = (
                    (p_leg[m_ord, 0] / np.sqrt(factorial))
                    * fac_sph(l_deg, m_ord)
                    * np.sqrt(2.0)
                )
                spherical_harm[i_sph, :] = common_fact * np.cos(m_ord * theta)
                i_sph += 1
                spherical_harm[i_sph, :] = common_fact * np.sin(m_ord * theta)
                # Update (l-m)!/(l+m)!
                factorial *= (l_deg + m_ord + 1.0) * (l_deg - m_ord)
            i_sph += 1
    return spherical_harm


def generate_derivative_real_spherical_harmonics(
    l_max: int, theta: np.ndarray, phi: np.ndarray
):
    r"""
    Generate derivatives of real spherical harmonics.

    If :math:`\phi` is zero, then the first component of the derivative wrt to
    :math:`phi` is set to zero.

    Parameters
    ----------
    l_max : int
        Largest angular degree of the spherical harmonics.
    theta : np.ndarray(N,)
        Azimuthal angle :math:`\theta \in [0, 2\pi]` that are being evaluated on.
        If this angle is outside of bounds, then periodicity is used.
    phi : np.ndarray(N,)
        Polar angle :math:`\phi \in [0, \pi]` that are being evaluated on.
        If this angle is outside of bounds, then periodicity is used.

    Returns
    -------
    ndarray(2, (l_max^2 + 1)^2, M)
        Derivative of spherical harmonics, (theta first, then phi) of all degrees up to
        :math:`l_{max}` and orders :math:`m` in Horton 2 order, i.e.
        :math:`m=0, 1, -1, \cdots, l, -l`.

    Notes
    -----
    - The derivative with respect to :math:`\phi` is

    .. math::
        \frac{\partial Y_l^m(\theta, \phi)}{\partial\phi} = -m Y_l^{-m}(\theta, \phi),
    - The derivative with respect to :math:`\theta` is

    .. math::
        \frac{\partial Y_l^m(\theta, \phi)}{\partial\phi} = |m| \cot(\phi)
         \Re (Y_l^{|m|}(\theta, \phi)) + \sqrt{(n - |m|)(n + |m| + 1)}
         Re(e^{-i\theta} Y_l^{|m| + 1}(\theta, \phi))
    for positive :math:`m` and for negative :math:`m`, the real projection :math:`\Re` is replaced
    with the imaginary projection :math:`\Im`.

    """
    num_pts = len(theta)
    # Shape (Derivs, Spherical, Pts)
    output = np.zeros((2, int((l_max + 1) ** 2), num_pts))

    complex_expon = np.exp(-theta * 1.0j)  # Needed for derivative wrt to phi
    l_list = np.arange(l_max + 1)
    sph_harm_vals = generate_real_spherical_harmonics(l_max, theta, phi)
    i_output = 0
    for l_val in l_list:
        for m in [0] + sum([[x, -x] for x in range(1, l_val + 1)], []):
            # Take all spherical harmonics at degree l_val
            sph_harm_degree = sph_harm_vals[(l_val) ** 2 : (l_val + 1) ** 2, :]

            # Take derivative wrt to theta :
            # for complex spherical harmonic it is   i m Y^m_l,
            # Note ie^(i |m| x) = -sin(|m| x) + i cos(|m| x), then take real/imaginery component.
            # hence why the negative is in (-m).
            # index_m maps m to index where (l, m)  is located in `sph_harm_degree`.
            index_m = lambda m: 2 * m - 1 if m > 0 else int(2 * np.fabs(m))
            output[0, i_output, :] = -m * sph_harm_degree[index_m(-m), :]

            # Take derivative wrt to phi:
            with np.errstate(divide="ignore", invalid="ignore"):
                cot_tangent = 1.0 / np.tan(phi)
            cot_tangent[np.abs(np.tan(phi)) < 1e-10] = 0.0
            # Calculate the derivative in two pieces:
            fac = np.sqrt((l_val - np.abs(m)) * (l_val + np.abs(m) + 1))
            output[1, i_output, :] = (
                np.abs(m) * cot_tangent * sph_harm_degree[index_m(m), :]
            )
            # Compute it using SciPy, removing conway phase (-1)^m and multiply by 2^0.5.
            sph_harm_m = (
                fac
                * sph_harm(np.abs(m) + 1, l_val, theta, phi)
                * np.sqrt(2)
                * (-1) ** m
            )
            if m >= 0:
                if m < l_val:  # When m == l_val, then fac = 0
                    output[1, i_output, :] += np.real(complex_expon * sph_harm_m)
            elif m < 0:
                # generate order m=negative real spherical harmonic
                if m > -l_val:
                    output[1, i_output, :] += np.imag(complex_expon * sph_harm_m)
            if m == 0:
                # sqrt(2.0) isn't included in Y_l^m only m \neq 0
                output[1, i_output, :] /= np.sqrt(2.0)
            i_output += 1
    return output


def solid_harmonics(l_max: int, sph_pts: np.ndarray):
    r"""
    Generate the solid harmonics from zero to a maximum angular degree.

    .. math::
        R_l^m(r, \theta, \phi) = \sqrt{\frac{4\pi}{2l + 1}} r^l Y_l^m(\theta, \phi)

    Parameters
    ----------
    l_max : int
        Largest angular degree of the spherical harmonics.
    sph_pts : ndarray(M, 3)
        Three-dimensional points in spherical coordinates :math:`(r, \theta, \phi)`, where
        :math:`r\geq 0, \theta \in [0, 2\pi]` and :math:`\phi \in [0, \pi]`.

    Returns
    -------
    ndarray((l_max + 1)^2, M)
        The solid harmonics from degree :math:`l=0=, \cdots, l_{max}` and for all orders :math:`m`,
        ordered as :math:`m=0, 1, -1, 2, -2, \cdots, l, -l` evaluated on :math:`M` points.

    """
    r, theta, phi = sph_pts.T
    spherical_harm = generate_real_spherical_harmonics(l_max, theta, phi)
    degrees = np.array(
        sum([[l_deg] * (2 * l_deg + 1) for l_deg in range(l_max + 1)], [])
    )
    return (
        spherical_harm
        * r ** degrees[:, None]
        * np.sqrt(4.0 * np.pi / (2 * degrees[:, None] + 1))
    )


def convert_derivative_from_spherical_to_cartesian(
    deriv_r, deriv_theta, deriv_phi, r, theta, phi
):
    r"""Convert the derivative form spherical coordinates to Cartesian.

    If :math:`r` is zero, then the derivative wrt to theta and phi are set to zero.
    If :math:`\phi` is zero, then the derivative wrt to theta is set to zero.

    Parameters
    ----------
    deriv_r, deriv_theta, deriv_phi : float, float, float
        Derivative wrt to spherical coordinates
    r, theta, phi : float, float, float
        The spherical points where the derivative is taken.

    Returns
    -------
    ndarray(3,) :
        The derivative wrt to (x, y, z) Cartesian coordinates.

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        jacobian = np.array(
            [
                [
                    np.cos(theta) * np.sin(phi),
                    -np.sin(theta) / (r * np.sin(phi)),
                    np.cos(theta) * np.cos(phi) / r,
                ],
                [
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta) / (r * np.sin(phi)),
                    np.sin(theta) * np.cos(phi) / r,
                ],
                [np.cos(phi), 0.0, -np.sin(phi) / r],
            ]
        )
    # If the radial component is zero, then put all zeros on the derivs of theta and phi
    if np.abs(r) < 1e-10:
        jacobian[:, 1] = 0.0
        jacobian[:, 2] = 0.0
    # If phi angle is zero, then set the derivative wrt to theta to zero
    if np.abs(phi) < 1e-10:
        jacobian[:, 1] = 0.0
    return jacobian.dot(np.array([deriv_r, deriv_theta, deriv_phi]))


def convert_cart_to_sph(points, center=None):
    r"""Convert a set of points from cartesian to spherical coordinates.

    Parameters
    ----------
    points : np.ndarray(n, 3)
        The (3-dimensional) Cartesian coordinates of points.
    center : np.ndarray(3,), list, optional
        Cartesian coordinates of the center of spherical coordinate system.
        If `None`, origin is used.

    Returns
    -------
    np.ndarray(N, 3)
        Spherical coordinates of atoms respect to the center
        [radius :math:`r`, azumuthal :math:`\theta`, polar :math:`\phi`]
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points array requires shape (N, 3), got: {points.ndim}")
    center = np.zeros(3, dtype=float) if center is None else np.asarray(center)
    if len(center) != 3:
        raise ValueError(f"center needs be of length (3), got:{center}")
    relat_pts = points - center
    # compute r
    r = np.linalg.norm(relat_pts, axis=-1)
    # polar angle: arccos(z / r)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.arccos(relat_pts[:, 2] / r)
    # fix nan generated when point is [0.0, 0.0, 0.0]
    phi[r == 0.0] = 0.0
    # azimuthal angle arctan2(y / x)
    theta = np.arctan2(relat_pts[:, 1], relat_pts[:, 0])
    return np.vstack([r, theta, phi]).T


def generate_orders_horton_order(order: int, type_ord: str, dim: int = 3):
    r"""
    Generate all orders from an integer :math:`l`.

    For Cartesian, the orders are :math:`(n_x, n_y, n_z)` such that they sum to `order`.
    If `dim=1,2`, then it generates Cartesian orders :math:`(n_x)`, :math:`(n_x, n_y)`,
    respectively, such that they sum to `order`.

    For radial, the orders is just the order :math:`l`.

    For spherical, the orders :math:`(l, m)` following the order
     :math:`[(l, 0), (l, 1), (l, -1), \cdots, (l, l), (l, -l)]`.

    Parameters
    ----------
    order: int
        The order :math:`l`.
    type_ord : str, optional
        The type of the order, it is either "cartesian", "radial" or "spherical".
    dim : int, optional
        The dimension of the orders for only Cartesian.

    Returns
    -------
    ndarray(3 * `order` , D)
        Each row is a list of `D` integers (e.g. :math:`(n_x, n_y, n_z)` or :math:`(l, m)`).
        The output is in Horton 2 order. e.g. order=2 it's
        [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], ....]

    """
    if not isinstance(order, int):
        raise TypeError(f"Order {type(order)} should be integer type.")
    if type_ord not in ["cartesian", "radial", "pure", "pure-radial"]:
        raise ValueError(f"Type {type_ord} is not recognized.")

    orders = []
    if type_ord == "cartesian":
        if dim == 3:
            for m_x in range(order, -1, -1):
                for m_y in range(order - m_x, -1, -1):
                    orders.append([m_x, m_y, order - m_x - m_y])
        elif dim == 2:
            for m_x in range(order, -1, -1):
                orders.append([m_x, order - m_x])
        elif dim == 1:
            return np.arange(0, order + 1, dtype=np.int)
        else:
            raise ValueError(f"dim {dim} parameter should be either 1, 2, 3.")
    elif type_ord == "radial":
        return np.array([order])
    elif type_ord == "pure":
        # Add the (l, 0)
        orders.append([order, 0])
        # Add orders (i, l) (i, -l) i=1, to l
        for x in range(1, order + 1):
            orders += [[order, x], [order, -x]]
    elif type_ord == "pure-radial":
        # Generate (n, l=0,1, 2 ..., (n-1), m=0, 1, -1, ... l -l)
        for l_deg in range(0, order):
            for m_ord in range(0, l_deg + 1):
                if m_ord != 0:
                    orders += [[order, l_deg, m_ord], [order, l_deg, -m_ord]]
                else:
                    orders += [[order, l_deg, m_ord]]
    else:
        raise ValueError(f"Type {type_ord} is not recognized.")
    orders = np.array(orders, dtype=int)
    return orders
