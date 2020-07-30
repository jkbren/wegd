"""
gdd.py
--------------------------

Graph diffusion distance, from

Hammond, D. K., Gur, Y., & Johnson, C. R. (2013, December). Graph diffusion
distance: A difference measure for weighted graphs based on the graph Laplacian
exponential kernel. In Global Conference on Signal and Information Processing,
2013 IEEE (pp 419-422). IEEE. https://doi.org/10.1109/GlobalSIP.2013.6736904

author: Brennan Klein
email: brennanjamesklein at gmail dot com
Submitted as part of the 2019 NetSI Collabathon.

"""

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import laplacian
from .base import BaseDistance


class GraphDiffusion(BaseDistance):
    """Find the maximally-dissimilar diffusion kernels between two graphs."""

    def dist(self, G1, G2, thresh=1e-14, resolution=1000):
        r"""
        The graph diffusion distance \cite{Hammond2013} between two graphs, $G$
        and $G'$, is a distance measure based on the notion of flow within each
        graph. As such, this measure uses the unnormalized Laplacian matrices
        of both graphs, $\mathcal{L}$ and $\mathcal{L}'$, and uses them to
        construct time-varying Laplacian exponential diffusion kernels,
        $e^{-t\mathcal{L}}$ and $e^{-t\mathcal{L}'}$, by effectively simulating
        a diffusion process for $t$ timesteps, creating a column vector of
        node-level activity at each timestep. The distance
        $d_\texttt{GDD}(G, G')$ is defined as the Frobenius norm between the
        two diffusion kernels at the timestep $t^{*}$ where the two kernels
        are maximally different.

        D_{GDD}(G,G') = \sqrt{||e^{-t^{*}\mathcal{L}}-e^{-t^{*}\mathcal{L}'}||}

        MATLAB code from: https://rb.gy/txbfrh

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared. Can also input np.arrays,
            both directed and undirected, weighted and unweighted.
            # NOTE: MATLAB IMPLEMENTATION REFERS TO THESE AS SYMMETRIC MATRICES

        thresh (float)
            minimum value above which the eigenvalues will be considered.

        resolution (int)
            number of t values to span through.

        Returns
        -------
        dist (float)
            the distance between G1 and G2.

        """

        if type(G1) == nx.Graph:
            A1 = nx.to_numpy_array(G1)
        if type(G2) == nx.Graph:
            A2 = nx.to_numpy_array(G2)

        L1 = laplacian(A1)
        L2 = laplacian(A2)

        D1, V1 = np.linalg.eig(L1)
        D2, V2 = np.linalg.eig(L2)

        eigs = np.hstack((np.diag(D1), np.diag(D2)))
        eigs = eigs[np.where(eigs > thresh)]
        eigs = np.sort(eigs)
        if len(eigs) == 0:
            dist = 0
            self.results['dist'] = dist
            return dist

        t_upperbound = np.real(1.0 / eigs[0])
        ts = np.linspace(0, t_upperbound, resolution)

        # Find the Frobenius norms between all the diffusion kernels at
        # different times. Return the value and where this vector is minimized.
        E = -gdd_xi_t(V1, D1, V2, D2, ts)
        f_val, t_star = (np.nanmin(E), np.argmin(E))

        dist = np.sqrt(-f_val)

        self.results['laplacian_matrix_1'] = L1
        self.results['laplacian_matrix_2'] = L2
        self.results['t_star'] = t_star
        self.results['f+_al'] = f_val

        self.results['dist'] = dist

        return dist


def gdd_xi_t(V1, D1, V2, D2, ts):
    """
    Computes frobenius norm of difference of laplacian exponential diffusion
    kernels, at specified timepoints.

    Parameters
    ----------
    V1, V2 (np.array)
        eigenvectors of the Laplacians of G1 and G2
    D1, D2 (np.array)
        eigenvalues of the Laplacians of G1 and G2
    t (float)
        time at wich to compute the difference in Frobenius norms

    Returns
    -------
    E (np.array)
        same size as t, contains differences of Frobenius norms

    """

    E = np.zeros(len(ts))

    for kt, t in enumerate(ts):
        ed1 = np.diag(np.exp(-t * np.diag(D1)))
        ed2 = np.diag(np.exp(-t * np.diag(D2)))

        tmp1 = V1.dot(np.atleast_2d(ed1).T * V1.T)
        tmp2 = V2.dot(np.atleast_2d(ed2).T * V2.T)
        tmp = tmp1 - tmp2

        E[kt] = sum(sum(tmp**2))

    return E
