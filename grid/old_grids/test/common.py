# -*- coding: utf-8 -*-
# OLDGRIDS: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The OLDGRIDS Development Team
#
# This file is part of OLDGRIDS.
#
# OLDGRIDS is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# OLDGRIDS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --


from contextlib import contextmanager
import hashlib
import os
import shutil
import subprocess
import tempfile

import numpy as np

from grid.cext import Cell
from grid.moments import get_cartesian_powers

__all__ = [
    'in_horton_source_root', 'check_script', 'check_script_in_tmp', 'check_delta',
    'get_random_cell', 'get_pentagon_moments', 'compare_occ_model', 'compare_orbs',
    'tmpdir', 'numpy_seed', 'truncated_file',
]

# All, except underflows, is *not* fine.
np.seterr(divide='raise', over='raise', invalid='raise')


def in_horton_source_root():
    """Test if the current directory is the OLDGRIDS source tree root"""
    # Check for some files and directories that must be present (for functions
    # that use this check).
    if not os.path.isfile('setup.py'):
        return False
    if not os.path.isdir('grid'):
        return False
    if not os.path.isdir('data'):
        return False
    if not os.path.isdir('scripts'):
        return False
    if not os.path.isfile('README'):
        return False
    with open('README') as f:
        if next(
                f) != 'OLDGRIDS: *H*elpful *O*pen-source *R*esearch *TO*ol for *N*-fermion systems.\n':
            return False
    return True


def check_script(command, workdir):
    """Change to workdir and try to run the given command.

       This can be used to test whether a script runs properly, with exit code
       0. When the example generates an output file, then use
       :py:function:`grid.test.common.check_script_in_tmp` instead.

       **Arguments:**

       command
            The command to be executed as a single string.

       workdir
            The work directory where the command needs to be executed.
    """
    env = dict(os.environ)
    root_dir = os.getcwd()
    env['PYTHONPATH'] = root_dir + ':' + env.get('PYTHONPATH', '')
    if in_horton_source_root():
        # This is only needed when running the tests from the source tree.
        env['OLDGRIDSDATA'] = os.path.join(root_dir, 'data')
    env['PATH'] = os.path.join(root_dir, 'scripts') + ':' + env.get('PATH', '')
    try:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=workdir, env=env, shell=True)
        outdata, errdata = proc.communicate()
    except OSError:
        raise AssertionError('Executable not found.')
    if proc.returncode != 0:
        print('Standard output')
        print(('+' * 80))
        print(outdata)
        print(('+' * 80))
        print('Standard error')
        print(('+' * 80))
        print(errdata)
        print(('+' * 80))
        assert False


def check_script_in_tmp(command, required, expected):
    """Test a script in a tmp dir

       **Arguments:**

       command
            The command to be executed in a tmp dir.

       required
            The required files to be copied to the tmp dir.

       expected
            A list of files expected to be present in the tmp dir after
            execution.
    """
    # Create a unique-ish suffix
    command = command.encode("utf-8")
    expected = [i.encode("utf-8") for i in expected]
    required = [i.encode("utf-8") for i in required]
    m = hashlib.sha256(command)
    for s in required + expected:
        m.update(s)
    suffix = m.hexdigest()
    # Do the actual work.
    with tmpdir('check_scrip_in_tmp-{}'.format(suffix).encode("utf-8")) as dn:
        # copy files into tmp
        for fn in required:
            shutil.copy(fn, os.path.join(dn, os.path.basename(fn)))
        # run the script
        check_script(command, dn)
        # check the output files
        for fn in expected:
            assert os.path.exists(os.path.join(dn, fn)), 'Missing output %s' % fn


def check_delta(fun, fun_deriv, x, dxs):
    """Check the difference between two function values using the analytical gradient

       Arguments:

       fun
            The function whose derivatives must be to be tested

       fun_deriv
            The implementation of the analytical derivatives

       x
            The argument for the reference point.

       dxs
            A list with small relative changes to x

       For every displacement in ``dxs``, the following computation is repeated:

       1) ``D1 = fun(x+dx) - fun(x)`` is computed.
       2) ``D2 = 0.5*dot(fun_deriv(x+dx) + fun_deriv(x), dx)`` is computed.

       A threshold is set to the median of the D1 set. For each case where |D1|
       is larger than the threshold, |D1 - D2|, should be smaller than the
       threshold.

       This test makes two assumptions:

       1) The gradient at ``x`` is non-zero.
       2) The displacements, ``dxs``, are small enough such that in the majority
          of cases, the linear term in ``fun(x+dx) - fun(x)`` dominates. Hence,
          sufficient elements in ``dxs`` should be provided for this test to
          work.
    """
    assert len(x.shape) == 1
    if len(dxs) < 20:
        raise ValueError('At least 20 displacements are needed for good statistics.')

    dn1s = []
    dn2s = []
    dnds = []
    f0 = fun(x)
    grad0 = fun_deriv(x)
    for dx in dxs:
        f1 = fun(x + dx)
        grad1 = fun_deriv(x + dx)
        grad = 0.5 * (grad0 + grad1)
        d1 = f1 - f0
        if hasattr(d1, '__iter__'):
            norm = np.linalg.norm
        else:
            norm = abs
        d2 = np.dot(grad, dx)

        dn1s.append(norm(d1))
        dn2s.append(norm(d2))
        dnds.append(norm(d1 - d2))
    dn1s = np.array(dn1s)
    dn2s = np.array(dn2s)
    dnds = np.array(dnds)

    # Get the threshold (and mask)
    threshold = np.median(dn1s)
    mask = dn1s > threshold
    # Make sure that all cases for which dn1 is above the treshold, dnd is below
    # the threshold
    if not (dnds[mask] < threshold).all():
        raise AssertionError((
                                 'The first order approximation on the difference is too wrong. The '
                                 'threshold is %.1e.\n\nDifferences:\n%s\n\nFirst order '
                                 'approximation to differences:\n%s\n\nAbsolute errors:\n%s')
                             % (threshold,
                                ' '.join('%.1e' % v for v in dn1s[mask]),
                                ' '.join('%.1e' % v for v in dn2s[mask]),
                                ' '.join('%.1e' % v for v in dnds[mask])
                                ))


def get_random_cell(a, nvec):
    if nvec == 0:
        return Cell(None)
    while True:
        if a <= 0:
            raise ValueError('The first argument must be strictly positive.')
        rvecs = np.random.uniform(0, a, (nvec, 3))
        cell = Cell(rvecs)
        if cell.volume > a ** nvec * 0.1:
            return cell


def get_pentagon_moments(rmat=None, lmax=4):
    if rmat is None:
        rmat = np.identity(3, float)

    cartesian_powers = get_cartesian_powers(lmax)
    ncart = cartesian_powers.shape[0]
    result = np.zeros(ncart)
    for i in range(6):
        alpha = 2.0 * np.pi / 5.0
        vec = np.array([1 + np.cos(alpha), np.sin(alpha), 0])
        vec = np.dot(rmat, vec)
        for j in range(ncart):
            px, py, pz = cartesian_powers[j]
            result[j] += vec[0] ** px * vec[1] ** py * vec[2] ** pz
    return result


def get_point_moments(coordinates, rmat=None, lmax=4):
    if rmat is None:
        rmat = np.identity(3, float)

    cartesian_powers = get_cartesian_powers(lmax)
    ncart = cartesian_powers.shape[0]
    result = np.zeros(ncart)
    for i in range(len(coordinates)):
        vec = np.dot(rmat, coordinates[i])
        for j in range(ncart):
            px, py, pz = cartesian_powers[j]
            result[j] += vec[0] ** px * vec[1] ** py * vec[2] ** pz
    return result


def compare_occ_model(occ_model1, occ_model2):
    assert occ_model1.__class__ == occ_model2.__class__
    if occ_model1 is None:
        assert occ_model2 is None
    elif hasattr(occ_model1, "nalpha") and (occ_model1, "nbeta"):
        assert occ_model1.nalpha == occ_model2.nalpha
        assert occ_model1.nbeta == occ_model2.nbeta
    else:
        raise NotImplementedError


def compare_orbs(orb1, orb2):
    assert orb1.nbasis == orb2.nbasis
    assert orb1.nfn == orb2.nfn
    assert (orb1.coeffs == orb2.coeffs).all()
    assert (orb1.energies == orb2.energies).all()
    assert (orb1.occupations == orb2.occupations).all()


@contextmanager
def tmpdir(name):
    dn = tempfile.mkdtemp(name)
    try:
        yield dn
    finally:
        shutil.rmtree(dn)


@contextmanager
def numpy_seed(seed=1):
    """Temporarily set NumPy's random seed to a given number.

    Parameters
    ----------
    seed : int
           The seed for NumPy's random number generator.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    yield None
    np.random.set_state(state)


@contextmanager
def truncated_file(name, fn_orig, nline, nadd):
    """Make a temporary truncated copy of a file.

    Parameters
    ----------
    name : str
           The name of test, used to make a unique temporary directory
    fn_orig : str
              The file to be truncated.
    nline : int
            The number of lines to retain.
    nadd : int
           The number of empty lines to add.
    """
    with tmpdir(name) as dn:
        fn_truncated = '%s/truncated_%i_%s' % (dn, nline, os.path.basename(fn_orig))
        with open(fn_orig) as f_orig, open(fn_truncated, 'w') as f_truncated:
            for counter, line in enumerate(f_orig):
                if counter >= nline:
                    break
                f_truncated.write(line)
            for _ in range(nadd):
                f_truncated.write('\n')
        yield fn_truncated
