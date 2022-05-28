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
    else:
        raise ValueError(f"Not supported radii type, got {type}")


def generate_real_spherical_harmonics(l_max, theta, phi):
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
        is stored, followed by positive orders then negative.

    Examples
    --------
    To obtain the l-th degree for all orders
    >>> spherical_harmonic = generate_real_spherical_harmonics(5, theta, phi)
    >>> desired_degree = 2
    >>> spherical_harmonic[(desired_degree)**2: (desired_degree + 1)**2, :]

    """
    if l_max < 0:
        raise ValueError(f"lmax needs to be >=0, got l_amx={l_max}")

    total_sph = np.zeros((0, len(theta)), dtype=float)
    l_list = np.arange(l_max + 1)
    for l_val in l_list:
        # generate m=0 real spheric
        zero_real_sph = sph_harm(0, l_val, theta, phi).real

        # generate order m=positive real spheric
        m_list_p = np.arange(1, l_val + 1, dtype=float)
        pos_real_sph = (
            sph_harm(m_list_p[:, None], l_val, theta, phi).real
            * np.sqrt(2)
            * (-1) ** m_list_p[:, None]
        )
        # generate order m=negative real spherical harmonic
        m_list_n = np.arange(-l_val, 0, dtype=float)
        neg_real_sph = (
            sph_harm(m_list_n[:, None], l_val, theta, phi).imag
            * np.sqrt(2)
            * (-1) ** m_list_n[:, None]
        )
        total_sph = np.vstack(
            (total_sph, zero_real_sph, pos_real_sph, neg_real_sph)
        )
    return total_sph


def convert_cart_to_sph(points, center=None):
    """Convert a set of points from cartesian to spherical coordinates.

    Parameters
    ----------
    points : np.ndarray(n, 3)
        3 dimentional numpy array for points
    center : np.ndarray(3,), list, optional
        center of the spherical coordinates
        [0., 0., 0.] will be used if `center` is not given

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
    phi = np.arccos(relat_pts[:, 2] / r)
    # fix nan generated when point is [0.0, 0.0, 0.0]
    phi[r == 0.0] = 0.0
    # azimuthal angle arctan2(y / x)
    theta = np.arctan2(relat_pts[:, 1], relat_pts[:, 0])
    return np.vstack([r, theta, phi]).T
