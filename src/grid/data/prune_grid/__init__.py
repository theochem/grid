r"""
Pre-defined atomic grid information in .npz file.

All have keys = atnum_rad or atnum_npt. atnum_rad for sg_1 correspond to the alpha parameter as
 they prune the grid based on regions defined by scaling the braag radii by each alpha.

Standard Grids
--------------
The "standard grids" are obtained from Q-GRID: `https://manual.q-chem.com/5.1/sect-quadrature.html`

"SG-0" is obtained from
.. [1] S.-H. Chien and P. M. W. Gill. J. Comput. Chem., 27:0 730, 2006. doi: rm10.1002/jcc.20383.

"SG-1" is from
.. [2] P. M. W. Gill, B. G. Johnson, and J. A. Pople. Chem. Phys. Lett., 209:0 506, 1993.
       doi:rm10.1016/0009-2614(93)80125-9.

"SG-2" and "SG-3" is from
.. [3] S. Dasgupta and J. M. Herbert. J. Comput. Chem., 38:0 869, 2017. doi:rm10.1002/jcc.24761.


Ochenfeld Grids
---------------

The Ochenfeld grids, denoted as g1, g2, g3, etc, are obtained from:

.. [4] Laqua, H., Kussmann, J., & Ochsenfeld, C. (2018). An improved molecular partitioning scheme
       for numerical quadratures in density functional theory. The Journal of Chemical Physics,
       149(20).

"""
