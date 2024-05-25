"""
sdes.distributions module:

We use this module to write custom extensions of the distributions that exist in the
particles.distributions module. These are constructed by subclassing the `ProbDist` class
for continuous random variables, and the `DiscreteDist` class for discrete random variables.

We also include a copy of the `VaryingCovNormal` class. This class exists in the latest version
of particles that is currently on Github, but not in the latest pip version: `0.3alpha` at the time of writing.
Use of this class is necessary to build the distribution objects for multivariate SDEs.
"""


from particles.distributions import MvNormal
import numpy as np
import numpy.linalg as nla
from scipy import stats

HALFLOG2PI = 0.5 * np.log(2.0 * np.pi)

class VaryingCovNormal(MvNormal):
    """Multivariate Normal (varying covariance matrix).

    Parameters
    ----------
    loc: ndarray
        location parameter (default: 0.)
    cov: (N, d, d) ndarray
        covariance matrix (no default)

    Notes
    -----

    Uses this distribution if you need to specify a Multivariate distribution
    where the covariance matrix varies across the N particles. Otherwise, see
    `MvNormal`.
    """
    def __init__(self, loc=0., cov=None):
        self.loc = loc
        self.cov = cov
        err_msg = 'VaryingCovNormal: argument cov must be a (N, d, d) array, \
                with d>1; cov[n, :, :] must be symmetric and positive'
        try:
            self.N, d1, d2 = self.cov.shape  # must be 3D
            self.L = nla.cholesky(self.cov)  # lower triangle
        except:
            raise ValueError(err_msg)
        assert d1 == d2, err_msg

    def linear_transform(self, z):
        return self.loc + np.einsum("...ij,...j", self.L, z)

    def rvs(self, size=None):
        N = self.N if size is None else size
        z = stats.norm.rvs(size=(N, self.dim))
        return self.linear_transform(z)

    def logpdf(self, x):
        halflogdetcov = np.sum(np.log(np.diagonal(self.L, axis1=1, axis2=2)),
                               axis=1)
        # not as efficient as triangular_solve, but numpy does not have
        # a "tensor" version of triangular_solve
        z = nla.solve(self.L, x - self.loc)
        norm_cst = self.dim * HALFLOG2PI + halflogdetcov
        return - 0.5 * np.sum(z * z, axis=1) - norm_cst

    def posterior(self, x, Sigma=None):
        raise NotImplementedError
