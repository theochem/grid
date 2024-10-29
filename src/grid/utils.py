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
"""Utility Module."""

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
        2.4944385,
        2.38105493,
        2.34326041,
        2.4944385,
        2.30546588,
        2.30546588,
        2.26767136,
        2.2487741,
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
        3.9117331,
        3.85504131,
        3.83614405,
        3.79834953,
        3.76055501,
        3.74165775,
        3.74165775,
        3.70386322,
        3.6660687,
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
        2.4944385,
        2.74010289,
        2.75900016,
        2.79679468,
        2.64561659,
        2.8345892,
        2.8345892,
        4.91328795,
        4.17629476,
        4.06291119,
        3.89283584,
        3.77945227,
        3.70386322,
        3.59047965,
        3.53378787,
        3.40150704,
        3.19363717,
    ]
)


# Obtained from https://www.chem.ualberta.ca/~massspec/atomic_mass_abund.pdf in atomic units
# [1] G. Audi, A. H. Wapstra Nucl. Phys A. 1993, 565, 1-65
# [2] G. Audi, A. H. Wapstra Nucl. Phys A. 1995, 595, 409-480.
isotopic_masses = {
    1: 1.007825,
    2: 4.002603,
    3: 7.016004,
    4: 9.012182,
    5: 11.009305,
    6: 12.0,
    7: 14.003074,
    8: 15.994915,
    9: 18.998403,
    10: 19.992440,
    11: 22.989770,
    12: 23.985042,
    13: 26.981538,
    14: 27.976927,
    15: 30.973762,
    16: 31.972071,
    17: 34.968853,
    18: 39.962383,
    19: 38.963707,
    20: 39.962591,
    21: 44.955910,
    22: 47.947947,
    23: 50.943964,
    24: 51.940512,
    25: 54.938050,
    26: 55.934942,
    27: 58.933200,
    28: 57.935348,
    29: 62.929601,
    30: 63.929147,
    31: 68.925581,
    32: 73.921178,
    33: 74.921596,
    34: 79.916522,
    35: 78.918338,
    36: 83.911507,
    37: 84.911789,
    38: 87.905614,
    39: 88.905848,
    40: 89.904704,
    41: 92.906378,
    42: 97.905408,
    43: 97.907216,
    44: 101.904350,
    45: 102.905504,
    46: 105.903483,
    47: 106.905093,
    48: 113.903358,
    49: 114.903878,
    50: 119.902197,
    51: 120.903818,
    52: 129.906223,
    53: 126.904468,
    54: 131.904154,
    55: 132.905447,
    56: 137.905241,
    57: 138.906348,
    58: 139.905434,
    59: 140.907648,
    60: 141.907719,
    61: 144.912744,
    62: 151.919728,
    63: 152.921226,
    64: 157.924101,
    65: 158.925343,
    66: 161.926795,
    67: 164.930319,
    68: 165.930290,
    69: 168.934211,
    70: 173.938858,
    71: 174.940768,
    72: 179.946549,
    73: 183.950933,
    74: 186.955751,
    75: 186.955751,
    76: 191.961479,
    77: 192.962924,
    78: 194.964774,
    79: 196.966552,
    80: 201.970626,
    81: 204.974412,
    82: 207.976636,
}

r"""
Obtained from theochem/horton/data/grids/tv-13.5.txt with the following preamble

These grids were created by Toon Verstraelen in July 2013 in a second effort
to generate tuned grids based on a series of diatomic benchmarks computed
with the QZVP basis of Ahlrichs and coworkers. They are constructed to give
on average 5 digits of precision for a variety of molecular integrals, using
the Becke scheme with k=3.
"""
# Items are (rmin, rmax, npts) in angstrom, Meant for PowerRTransform
_DEFAULT_POWER_RTRANSFORM_PARAMS = {
    1: (2.577533167224667e-07, 16.276983371222354, 34),
    2: (3.2755036736843646e-07, 20.13904927593614, 34),
    3: (6.186114573134926e-09, 18.12176164518007, 71),
    4: (1.4059661477491232e-08, 15.44201607973657, 59),
    5: (3.273788764659501e-08, 13.56903494126234, 49),
    6: (3.11827230204136e-08, 34.864731811979844, 59),
    7: (3.401062176724264e-08, 14.764587345673986, 49),
    8: (1.3926503705988068e-08, 15.016481365384685, 59),
    9: (1.1147270392375693e-08, 12.748095827643704, 59),
    10: (1.6588702526298265e-08, 18.151260320096828, 59),
    11: (3.003120602822434e-09, 21.596666254555863, 85),
    12: (5.9819613661137094e-09, 16.828409532166827, 71),
    13: (7.58358121259898e-09, 21.572146722677854, 71),
    14: (8.373611591213212e-09, 23.96629707958869, 71),
    15: (2.297851714218129e-09, 16.418299079716235, 85),
    16: (3.1813786400179616e-09, 22.79566890037017, 85),
    17: (6.578265835890989e-09, 18.74786834668225, 71),
    18: (2.7411385440998967e-09, 19.555180372886927, 85),
    19: (1.089946590791849e-09, 20.952936209264042, 103),
    20: (1.167544045884533e-09, 22.376677702840254, 103),
    21: (1.10777797534853e-09, 21.260508469961408, 103),
    22: (2.746202778795927e-09, 19.350116266638185, 85),
    23: (2.9402748888394357e-09, 21.39933482952019, 85),
    24: (2.7579341642634884e-09, 19.83428261878494, 85),
    25: (2.506969959804919e-09, 17.92147397464149, 85),
    26: (2.523212584292888e-09, 18.113798778890985, 85),
    27: (4.744390728782565e-09, 34.30270780321776, 85),
    28: (2.350841751293044e-09, 16.932289049448002, 85),
    29: (9.843092065564594e-10, 18.888340007851557, 103),
    30: (1.0768549234739597e-09, 20.652740486523665, 103),
    31: (6.188017478991146e-10, 29.4174935749708, 123),
    32: (3.055246063471103e-10, 37.391722877573585, 148),
    33: (4.808816730665759e-10, 22.858274814681398, 123),
    34: (6.632348945772076e-11, 16.198268042509657, 71),
    35: (4.3301047214875017e-14, 16.309446059148527, 85),
    36: (1.9525491942801685e-10, 23.897989168467937, 148),
    37: (1.3116816051165964e-05, 25.107553485889394, 85),
    38: (0.00012337243816711258, 20.15779186049731, 71),
    39: (0.00026254231985602306, 18.42725236767048, 59),
    40: (0.00022729864142629063, 19.240323445476516, 59),
    41: (0.00022332978749485058, 18.43051033247378, 59),
    42: (4.312335817136105e-05, 18.796698566238717, 59),
    43: (0.0001439045024590461, 18.262366596363997, 49),
    44: (0.0002705098774739686, 15.620015640557597, 49),
    45: (3.666445913651392e-05, 16.50582512555016, 49),
    46: (3.644416354245693e-05, 16.89283645207093, 59),
    47: (4.076886839028324e-06, 15.848411073664625, 59),
    48: (6.970977432467025e-07, 22.788090976720063, 71),
    49: (6.295388435898326e-07, 16.528521880671736, 59),
    50: (5.737464545147426e-08, 15.957552298428514, 71),
    51: (4.2786625564460944e-08, 16.134070100096796, 71),
    52: (2.92986601404113e-08, 15.709608688491917, 71),
    53: (2.6372540021175668e-08, 15.904602040581757, 71),
    54: (7.905521967940802e-09, 19.09928852814914, 71),
    55: (0.001011462489909521, 24.757734721809108, 71),
    56: (0.00038862580532087804, 22.298081065756328, 71),
    57: (1.760652355460472e-05, 21.958172637007983, 71),
    72: (0.0008857694481690485, 17.06721283130558, 59),
    73: (0.0010898387425983486, 22.98828587346505, 59),
    74: (0.0020116892152010754, 17.704162744894614, 49),
    75: (0.001554112459865995, 17.462135046822063, 49),
    76: (0.002694590130905911, 17.015182182896186, 49),
    77: (0.0036410660971326883, 20.76895835571112, 59),
    78: (0.005019006440323396, 17.81498800521723, 49),
    79: (0.0009784761183226729, 15.080040644442699, 49),
    80: (2.0642398272082567e-05, 20.018841788331276, 49),
    81: (4.213213544795557e-06, 20.909208619831304, 71),
    82: (2.5579357266911953e-06, 16.70520497941137, 59),
}


def get_cov_radii(atnums, cov_type="bragg"):
    """
    Get the covalent radii for given atomic number(s).

    Parameters
    ----------
    atnums : int or np.ndarray
        Atomic number(s) of the element(s) of interest.
    cov_type : str, optional
        Type of covalent radii for elements. Possible values are:
        - "bragg": Bragg-Slater empirically measured covalent radii.
        - "cambridge": Covalent radii from analysis of the Cambridge Database.
        - "alvarez": Covalent radii from https://doi.org/10.1039/B801115J.

    Returns
    -------
    np.ndarray
        Covalent radii of the desired atom(s) in atomic units.

    """
    if isinstance(atnums, (int, np.integer)):
        atnums = np.array([atnums])
    if np.any(np.array(atnums) == 0):
        raise ValueError(f"0 is not a valid atomic number, got {atnums}")
    if cov_type == "bragg":
        return _bragg[atnums]
    if cov_type == "cambridge":
        return _cambridge[atnums]
    if cov_type == "alvarez":
        return _alvarez[atnums]
    raise ValueError(f"Not supported covalent radii type, got {cov_type}")


def generate_real_spherical_harmonics_scipy(l_max: int, theta: np.ndarray, phi: np.ndarray):
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
        Value of real spherical harmonics of all orders :math:`m`, and degree
        :math:`l` spherical harmonics. For each degree :math:`l`,
        the orders :math:`m` are in HORTON 2 order, i.e.
        :math:`m=0, 1, -1, 2, -2, \cdots, l, -l`.

    Notes
    -----
    - SciPy spherical harmonics is known (Jan 30, 2024) to give NaN when the degree is large,
      for our experience, when l >= 86.
    """
    if l_max < 0:
        raise ValueError(f"lmax should be non-negative, got l_amx={l_max}")

    total_sph = np.zeros((0, len(theta)), dtype=float)
    l_list = np.arange(l_max + 1)
    for l_val in l_list:
        # generate m=0 real spherical harmonic
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
        total_sph = np.vstack((total_sph, zero_real_sph, *horton_ord))
    return total_sph


def generate_real_spherical_harmonics(l_max: int, theta: np.ndarray, phi: np.ndarray):
    r"""
    Compute the real spherical harmonics recursively up to a maximum angular degree l.

    .. math::
        Y_l^m(\theta, \phi) = \sqrt{\frac{(2l + 1) (l - m)!}{4\pi (l + m)!}} f(m, \theta)
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
    sin_phi = np.sin(phi, dtype=np.longdouble)
    cos_phi = np.cos(phi, dtype=np.longdouble)
    spherical_harm = np.zeros(((l_max + 1) ** 2, numb_pts), dtype=np.longdouble)

    # Forward recursion requires P_{l-1}^m, P_{l-2}^m, these are the two columns, respectively
    # the rows are the order m which ranges from 0 to l_max and p_leg[:l, :] gets updated every l
    p_leg = np.zeros((l_max + 1, 2, numb_pts), dtype=np.longdouble)
    p_leg[0, :, :] = 1.0  # Initial conditions: P_0^0 = 1.0

    # the coefficients of the forward recursions and initial factor of spherical harmonic.
    def a_k(deg, ord):
        return (2.0 * (float(deg) - 1.0) + 1) / (float(deg) - 1.0 - float(ord) + 1.0)

    def b_k(deg, ord):
        return (float(deg) - 1.0 + float(ord)) / (float(deg) - float(ord))

    def fac_sph(deg, ord):
        return np.sqrt((2.0 * float(deg) + 1) / (4.0 * np.pi))  # Note (l-m)!/(l+m)! is moved

    # Go through each degree and then order and fill out
    spherical_harm[0, :] = fac_sph(0.0, 0.0)  # Compute Y_0^0
    i_sph = 1  # Index to start of spherical_harm
    for l_deg in np.arange(1, l_max + 1, dtype=int):
        for m_ord in np.arange(0, l_deg + 1, dtype=int):
            if l_deg == m_ord:
                # Do diagonal spherical harmonic Y_l^m, when l=m.
                # Diagonal recursion: P_m^m = sin(phi) * P_{m-1}^{m-1} * (2 (l - 1) + 1)
                p_leg[m_ord, 0] = p_leg[m_ord - 1, 1] * (2 * (l_deg - 1.0) + 1) * sin_phi
            else:
                # Do forward recursion here and fill out Y_l^m and Y_l^{-m}
                # Compute b_k P_{l-2}^m,  since m < l, then m < l - 2
                second_fac = b_k(l_deg, m_ord) * p_leg[m_ord, 1] if m_ord <= l_deg - 2 else 0.0
                # Update/re-define P_{l-2}^m to be equal to P_{l-1}^m
                p_leg[m_ord, 1, :] = p_leg[m_ord, 0]
                # Then update P_{l-1}^m := P_l^m := a_k cos(\phi) P_{l-1}^m - b_k P_{l, -2}^m
                p_leg[m_ord, 0, :] = a_k(l_deg, m_ord) * cos_phi * p_leg[m_ord, 0] - second_fac
            # Compute Y_l^{m} that has cosine(theta) and Y_l^{-m} that has sin(theta)
            if m_ord == 0:
                # init factorial needed to compute (l-m)!/(l+m)!
                #  Turn the number into an array of longdouble type because of Overflow error
                #  for high degrees, also add the square-root to mitigate it too
                factorial = np.sqrt(np.array([(l_deg + 1.0) * l_deg], dtype=np.longdouble))
                spherical_harm[i_sph, :] = fac_sph(l_deg, m_ord) * p_leg[m_ord, 0]
            else:
                common_fact = (
                    (p_leg[m_ord, 0] / factorial[0]) * fac_sph(l_deg, m_ord) * np.sqrt(2.0)
                )
                spherical_harm[i_sph, :] = common_fact * np.cos(float(m_ord) * theta)
                i_sph += 1
                spherical_harm[i_sph, :] = common_fact * np.sin(float(m_ord) * theta)
                # Update (l-m)!/(l+m)!
                factorial[0] *= np.sqrt(
                    (float(l_deg) + float(m_ord) + 1.0) * (float(l_deg) - float(m_ord))
                )
            i_sph += 1
    return spherical_harm


def generate_derivative_real_spherical_harmonics(l_max: int, theta: np.ndarray, phi: np.ndarray):
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
    output = np.zeros((2, int((l_max + 1) ** 2), num_pts), dtype=np.longdouble)

    complex_expon = np.exp(-theta * 1.0j)  # Needed for derivative wrt to phi
    l_list = np.arange(l_max + 1)
    sph_harm_vals = generate_real_spherical_harmonics(l_max, theta, phi)
    i_output = 0
    for l_val in l_list:
        for m in [0, *sum([[x, -x] for x in range(1, l_val + 1)], [])]:
            # Take all spherical harmonics at degree l_val
            sph_harm_degree = sph_harm_vals[(l_val) ** 2 : (l_val + 1) ** 2, :]

            # Take derivative wrt to theta :
            # for complex spherical harmonic it is   i m Y^m_l,
            # Note ie^(i |m| x) = -sin(|m| x) + i cos(|m| x), then take real/imaginery component.
            # hence why the negative is in (-m).
            # index_m maps m to index where (l, m)  is located in `sph_harm_degree`.
            def index_m(m):
                return 2 * m - 1 if m > 0 else int(2 * np.fabs(m))

            output[0, i_output, :] = -float(m) * sph_harm_degree[index_m(-m), :]

            # Take derivative wrt to phi:
            with np.errstate(divide="ignore", invalid="ignore"):
                cot_tangent = 1.0 / np.tan(phi)
            cot_tangent[np.abs(np.tan(phi)) < 1e-10] = 0.0
            # Calculate the derivative in two pieces:
            fac = np.sqrt((l_val - np.abs(float(m))) * (l_val + np.abs(m) + 1))
            output[1, i_output, :] = np.abs(float(m)) * cot_tangent * sph_harm_degree[index_m(m), :]
            # Compute it using SciPy, removing conway phase (-1)^m and multiply by 2^0.5.
            sph_harm_m = (
                fac
                * sph_harm(np.abs(float(m)) + 1, l_val, theta, phi)
                * np.sqrt(2)
                * (-1.0) ** float(m)
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
        sum([[float(l_deg)] * (2 * l_deg + 1) for l_deg in np.arange(l_max + 1, dtype=int)], []),
        dtype=np.longdouble,
    )
    return (
        spherical_harm * r ** degrees[:, None] * np.sqrt(4.0 * np.pi / (2 * degrees[:, None] + 1))
    )


def convert_derivative_from_spherical_to_cartesian(deriv_r, deriv_theta, deriv_phi, r, theta, phi):
    r"""
    Convert the derivative from spherical coordinates to Cartesian.

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

    The convention that :math:`\theta \in [-\pi, \pi]` and :math:`\phi \in [0, \pi)`
    is chosen.

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

    For pure, the orders :math:`(l, m)` following the order
    :math:`[(l, 0), (l, 1), (l, -1), \cdots, (l, l), (l, -l)]`.

    Parameters
    ----------
    order: int
        The order :math:`l`.
    type_ord : str, optional
        The type of the order, it is either "cartesian", "radial", "pure" or "pure-radial".
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


def dipole_moment_of_molecule(grid, density: np.ndarray, coords: np.ndarray, charges: np.ndarray):
    r"""
    Calculate the dipole of a molecule.

    This is defined as the observable form a wavefunction :math:`\Psi`:

    .. math::
        \vec{\mu} = \int \Psi \hat{\mu} \Psi \vec{r}

    which results in

    .. math::
        \vec{\mu} = \sum_{i=1}^{N_{atoms}} Z_i (\vec{R_i} - \vec{R_c}) -
        \int ((\vec{r} - \vec{R_c})) \rho(\vec{r}) dr,

    where :math:`N_{atoms}` is the number of atoms, :math:`Z_i` is the atomic charge of the
    ith atom, :math:`\vec{R_i}` is the ith coordinate of the atom, :math:`\vec{R_c}` is the
    center of mass of the molecule in atomic units and :math:`\rho` is the electron density
    of the molecule.

    Parameters
    ----------
    grid: Grid
        The grid used to perform the integration.
    density: ndarray
        The electron density evaluated on points on `grid` object.
    coords: ndarray[M, 3]
        The coordinates of the atoms.
    charges: ndarray[M]
        The atomic charge of each atom.

    Returns
    -------
    ndarray:
        Returns array of size three corresponding to the dipole moment of a molecule.

    """
    # Calculate the center of mass of the molecule
    masses = np.array([isotopic_masses[charge] for charge in charges])
    center_mol = np.array([np.sum(coords * masses[:, None], axis=0) / np.sum(masses)])

    # Calculate the Cartesian moments of the electron density
    #     orders should be [[n_x, n_y, n_z]] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    #     integrals is \int \rho (x - X_c)^{n_x} (y - y_C) ^{n_y} (z - Z_c)^{n_x} dx dy dz
    integrals, orders = grid.moments(
        1, center_mol, density, type_mom="cartesian", return_orders=True
    )

    # calculate (X_a - X_c)**{n_x} (Y_a - Y_c)**{n_y} (Z_a - Z_c)**{n_z}
    cent_pts_with_order = (coords - center_mol) ** orders[:, None]
    # multiply over each corresponding moment axis (row)
    cent_pts_with_order = np.prod(cent_pts_with_order, axis=2)
    # calculate Z_a (X_a - X_c)**{n_x} (Y_a - Y_c)**{n_y} (Z_a - Z_c)**{n_z}
    result = np.einsum("ij,j->i", cent_pts_with_order, charges)

    # sum electric dipole moment and remove [0,0,0] moment value
    result = (result - integrals.T).flatten()[1:]

    return result
