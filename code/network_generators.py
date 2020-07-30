# network_generators.py
import numpy as np
import networkx as nx
from scipy.optimize import fsolve


def erdos_renyi_graph(n, k_avg):
    r"""
    Generates an Erdos-Renyi random graph by randomly connecting two nodes, $i$
    and $j$, with a probability $p$, corresponding to the specified average
    degree, $\langle k \rangle$.

    Parameters
    ----------
    n (int): number of nodes
    k_avg (float): desired average degreee of the resulting network

    Returns
    -------
    g (nx.Graph): a networkx graph

    """

    a = np.triu(np.random.rand(n, n) < k_avg / (n - 1), 1)
    g = nx.from_numpy_array(np.array(a + a.T, dtype=int))

    return g


def random_geometric_graph(n, k_avg):
    r"""
    Generates a random geometric graph by randomly placing nodes at coordinates
    on a 1-dimensional ring, then connecting nodes within a given radius.

    Note: by placing nodes on a 1-dimensional ring, this ensures that the
          density of random geometric graphs from this formulation has the same
          interpretation as the density of Erdos-Renyi random graphs.

    Parameters
    ----------
    n (int): number of nodes
    k_avg (float): desired average degreee of the resulting network

    Returns
    -------
    g (nx.Graph): networkx random geometric graph

    """

    x = np.random.rand(n)
    xo = np.outer(x, np.ones(n))
    ox = xo.T
    di = 0.5 - np.abs(0.5 - np.abs(ox - xo))
    R = 0.5 * k_avg / (n - 1)
    a = np.array(di <= R, dtype=int) - np.eye(n)
    g = nx.from_numpy_array(a)

    return g


def powerlaw_configuration_model(n, gamma, k_avg, discretize=True, nxout=True):
    r"""
    Generates a configuration model under a certain powerlaw exponent
    specification, size, and average degree.

    Parameters
    ----------
    n (int): size of the desired network
    gamma (float): value of the powerlaw exponent. Must be greater than 1.0 and
                   gamma = 3.0 will create the same degree distribution as a
                   BA network.
    k_avg (float): average degree of the desired network
    discretized (bool): should the degree sequence be discretized
    nxout (bool): return an adjacency matrix or a networkx graph object

    Returns
    -------
    adjacency_matrix (np.ndarray): the resulting configuration model network,
                                   either as a numpy array or as a networkx
                                   Graph object, depending on nxout.

    """

    if gamma <= 2.0:
        gamma = 2.00001

    degree_sequence = powerlaw_degree_sequence(n, gamma, k_avg, discretize)
    degree_sequence[degree_sequence > n-1] = n-1
    degree_sequence[np.argmax(degree_sequence)] -= sum(degree_sequence) % 2

    if nxout:
        return nx.configuration_model(degree_sequence, create_using=nx.Graph())

    else:
        return nx.to_numpy_array(nx.configuration_model(degree_sequence))


def powerlaw_degree_sequence(n, gamma, k_avg, discretize=True):
    r"""
    Generate a powerlaw degree sequence, which is commonly used to construct
    configuration model networks of a given graph.

    Parameters
    ----------
    n (int): size of the desired network
    gamma (float): value of the powerlaw exponent. Must be greater than 1.0 and
                   gamma = 3.0 will create the same degree distribution as a
                   BA network.
    k_avg (float): average degree of the desired network
    discretized (bool): should the degree sequence be discretized

    Returns
    -------
    degree_sequence (np.ndarray): a vector of degrees for the nodes of a
                                  network, in order of the nodes in the network

    """

    if gamma <= 2.0:
        gamma = 2.0001

    uniform_sample = np.random.rand(n)
    k_min = k_avg * (gamma - 2.0) / (gamma - 1.0)
    degree_sequence = k_min * ((1.0 - uniform_sample)**(-1.0 / (gamma - 1.0)))

    if discretize:
        degree_sequence = np.array(degree_sequence, dtype=int)

    return degree_sequence


# def erased_configuration_model(degree_sequence):
#     r"""
#     Generate an adjaceny matrix corresponding to an erased configuration model,
#     which is a configuration model from a degree sequence where the loops and
#     multi-edges are erased.

#     Parameters
#     ----------
#     degree_sequence (np.ndarray): a vector of degree values corresponding to a
#                                 desired per-node degree in a resulting network.

#     Returns
#     -------
#     A (np.ndarray): adjacency matrix of the corresponding configuration model.

#     """

#     n = len(degree_sequence)
#     edge_list = []

#     degree_list = sum([[j] * degree_sequence[j]
#                       for j in range(len(degree_sequence))], [])

#     for _ in range(int(sum(degree_sequence)/2)):
#         s, t = np.random.choice(degree_list, size=2, replace=False)

#         edge_list.append((s, t))
#         degree_list.remove(s)
#         degree_list.remove(t)

#     A = np.zeros((n, n))

#     for eij in edge_list:

#         if eij[0] != eij[1]:
#             A[eij[0], eij[1]] = 1
#             A[eij[1], eij[0]] = 1

#     return A

def erased_configuration_model(n, gamma, k_avg, discretize=True):
    r"""
    Generate an adjaceny matrix corresponding to an erased configuration model,
    which is a configuration model from a degree sequence where the loops and
    multi-edges are erased.

    Parameters
    ----------
    degree_sequence (np.ndarray): a vector of degree values corresponding to a
                                desired per-node degree in a resulting network.

    Returns
    -------
    A (np.ndarray): adjacency matrix of the corresponding configuration model.

    """
    if gamma <= 2.0:
        gamma = 2.00001

    degree_sequence = powerlaw_degree_sequence(n, gamma, k_avg, discretize)
    # degree_sequence[degree_sequence > n-1] = n-1

    n = len(degree_sequence)
    edge_list = []

    degree_list = sum([[j] * degree_sequence[j]
                      for j in range(len(degree_sequence))], [])

    for _ in range(int(sum(degree_sequence)/2)):
        s, t = np.random.choice(degree_list, size=2, replace=False)

        edge_list.append((s, t))
        degree_list.remove(s)
        degree_list.remove(t)

    A = np.zeros((n, n))

    for eij in edge_list:

        if eij[0] != eij[1]:
            A[eij[0], eij[1]] = 1
            A[eij[1], eij[0]] = 1

    return nx.from_numpy_array(A)


def soft_configuration_model(n, k_avg, gamma, discretize=True):
    r"""
    The Soft Configuration Model is the maximum-entropy random graph of a given
    sequence of expected degrees.

    Parameters
    ----------
    n (int): size of the desired network
    k_avg (float): average degree of the desired network
    gamma (float): value of the powerlaw exponent. Must be greater than 1.0 and
                   gamma = 3.0 will create the same degree distribution as a
                   BA network.
    discretize (bool): should the degree sequence be discretized

    Returns
    -------
    G (nx.Graph): adjacency matrix of the graph generated under this soft
                  configuration model.

    """

    if gamma <= 2.0:
        gamma = 2.00001

    eds = powerlaw_degree_sequence(n, gamma, k_avg, discretize)

    n = len(eds)

    f = lambda L: np.sum(1.0 / (np.outer(np.exp(L), np.exp(L))), axis=1) - eds
    L = fsolve(f, np.random.rand(n), xtol=10e-5)

    eL = np.exp(L)
    p = 1.0/(np.outer(eL, eL)+1.0)
    P = p-np.diag(np.diag(p))
    a = np.triu(np.array(np.random.rand(n, n) <= P, dtype=int), k=1)

    A = a + a.transpose()

    G = nx.from_numpy_array(A, create_using=nx.Graph)

    return G


def dorogovtsev_mendes_samukhin(N=100, k_avg=8, gamma=3.5):
    r"""
    The Dorogovtsev Mendes Samukhin (also sometimes referred to as the
    shifted linear attachment model or 'initial attractiveness' model) is a
    network growth model where the degree distribution of the resulting graph
    is tune-able by the initial attractiveness  of the nodes in the network.
    Note: multi-edges are not permitted in this model.

    This function is derived from the following paper:

    Dorogovtsev, S. N., Mendes, J. F. F., & Samukhin, A. N. (2000).
    Structure of growing networks with preferential linking.
    Physical review letters, 85(21), 4633.

    Parameters
    ----------
    N (int): number of nodees in the network, must be greater than n0
    k_avg (int): average degree of the final graph, which in this case should
                 be an even integer.
    gamma (float): any positive number greater than 2.0
    n0 (int): initial number of nodes in the graph. Note: n0 must be greater
              than $m$, which is int(k_avg/2).

    Returns
    -------
    G (nx.Graph): the resulting shifted linear attachment graph.

    """

    if gamma <= 2.0:
        gamma = 2.00001

    m = int(k_avg / 2)
    A = m * (gamma - 3)  # initial attractiveness value, not adjacency matrix.

    G = nx.complete_graph(m+1)

    for node_i in range(m+1, N):
        degrees = A + np.array(list(dict(G.degree()).values()))
        probs = (degrees) / sum(degrees)
        eijs = np.random.choice(
                    G.number_of_nodes(), size=(m,),
                    replace=False, p=probs)
        for node_j in eijs:
            G.add_edge(node_i, node_j)

    return G


def isotropic_redirection(n):
    r"""
    The Isotropic Redirection (or certain redirection) model is a simple
    network growth model wherein a new node joins the network by selecting
    a node at random, then selecting randomly one of that node's neighbors,
    and attaching a single link to this new node.

    References:
        Krapivsky, P. L., & Redner, S. (2001).
        Organization of growing random networks.
        Physical Review E, 63(6), 066123.

        and

        Krapivsky, P. L., & Redner, S. (2017).
        Emergent network modularity.
        Journal of Statistical Mechanics: Theory and Experiment, (7), 073405.

    Parameters
    ----------
    n (int): the size of the network

    Returns
    -------
    G (nx.Graph): the resulting isotropic redirection graph

    """

    adj = {0: [1], 1: [0]}

    for t in range(2, n):
        # pick a random existing node
        a = np.random.randint(t)

        # pick a random neighbor
        b = np.random.choice(adj[a])

        # connect to the neighbor
        adj[b].append(t)

        adj[t] = [b]

    el = [(i, j) for i in adj.keys() for j in adj[i]]
    G = nx.from_edgelist(el, create_using=nx.Graph)

    return G


def preferential_attachment_network(N, alpha=1.0, m=1):
    r"""
    Generates a network based off of a preferential attachment growth rule.
    Under this growth rule, new nodes place their $m$ edges to nodes already
    present in the graph, G, with a probability proportional to $k^\alpha$.

    Parameters
    ----------
    N (int): the desired number of nodes in the final network
    alpha (float): the exponent of preferential attachment. When alpha is less
                   than 1.0, we describe it as sublinear preferential
                   attachment. At alpha > 1.0, it is superlinear preferential
                   attachment. And at alpha=1.0, the network was grown under
                   linear preferential attachment, as in the case of
                   Barabasi-Albert networks.
    m (int): the number of new links that each new node joins the network with.

    Returns
    -------
    G (nx.Graph): a graph grown under preferential attachment.

    """

    G = nx.Graph()
    G = nx.complete_graph(m+1)

    for node_i in range(m+1, N):
        degrees = np.array(list(dict(G.degree()).values()))
        probs = (degrees**alpha) / sum(degrees**alpha)
        eijs = np.random.choice(
                    G.number_of_nodes(), size=(m,),
                    replace=False, p=probs)
        for node_j in eijs:
            G.add_edge(node_i, node_j)

    return G
