# from functools import partial
import warnings

import numpy as np

from scipy.special import sph_harm
from scipy.interpolate import CubicSpline


def generate_real_sph_harms(max_l, theta, phi):
    sph_h = generate_sph_harms(max_l, theta, phi)
    return np.nan_to_num(_convert_ylm_to_zlm(sph_h))


def generate_sph_harms(max_l, theta, phi):
    # theta azimuthal, phi polar
    l, m = _generate_sph_paras(max_l)
    return sph_harm(m[:, None, None], l[None, :, None], theta, phi)


def _generate_sph_paras(max_l):
    l_list = np.arange(max_l + 1)
    m_list_p = np.arange(max_l + 1)
    m_list_n = np.arange(-max_l, 0)
    m_list = np.append(m_list_p, m_list_n)
    return l_list, m_list


# def _prepare_sph_harms(l_list, m_list):
#     return partial(sph_harm, m=m_list[:, None, None], n=l_list[None, :, None])


# def _calculate_sph_harms(theta, phi, max_l):
#     # theta: Azimuthal [0, 2pi]
#     # phi: Polar [0, pi]
#     sph_func = _prepare_sph_harms(*_generate_sph_paras(max_l))
#     return sph_func(theta, phi)


def _convert_ylm_to_zlm(sp_harm_arrs):
    ms, ls, arrs = sp_harm_arrs.shape  # got list of Ls, and Ms
    # ls = l_max + 1
    # ms = 2 * l_max + 1
    s_h_r = np.zeros((ms, ls, arrs))  # copy old array for construct real
    # silence cast warning for complex -> float
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_h_r[1:ls] = (sp_harm_arrs[1:ls] + np.conjugate(sp_harm_arrs[1:ls])) / np.sqrt(
            2
        )
        s_h_r[-ls + 1 :] = (
            sp_harm_arrs[-ls + 1 :] - np.conjugate(sp_harm_arrs[-ls + 1 :])
        ) / (np.sqrt(2) * 1j)
        s_h_r[0] = sp_harm_arrs[0]
    return s_h_r


# def atomic_sph_harm_values(atomic_grid, max_l):
#     # compute theta and phi
#     atomic_grid.convert_cart_to_sph()
#     theta, phi = atomic_grid.sph_grid
#     return calculate_sph_harms(theta, phi, max_l)


def condense_values_with_sph_harms(sph_harm, value_arrays, weights, indices, radial):
    prod_value = sph_harm * value_arrays * weights
    total = len(indices) - 1
    axis_value = []
    for i in range(total):
        sum_i = np.sum(prod_value[:, :, indices[i] : indices[i + 1]], axis=-1)
        axis_value.append(sum_i)
    ml_sph_value = np.nan_to_num(np.stack(axis_value))
    return CubicSpline(x=radial, y=ml_sph_value)


def interpelate(spline, r_point, theta, phi):
    r_value = spline(r_point)
    l_max = r_value.shape[1] - 1
    r_sph_harm = generate_real_sph_harms(l_max, theta, phi)
    return np.sum(r_sph_harm * r_value[:, :, None], axis=(0, 1)) / (r_point ** 2)
