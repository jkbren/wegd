from network_generators import erdos_renyi_graph
import numpy as np
import networkx as nx
import sys
from netrd.distance import *
from graph_diffusion import GraphDiffusion
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def LJ(G1, G2):
    D = LaplacianSpectral()
    d = D.dist(G1, G2, kernel='lorentzian')
    return d


def LE(G1, G2):
    D = LaplacianSpectral()
    d = D.dist(G1, G2, kernel='lorentzian', measure='euclidean')
    return d


def NJ(G1, G2):
    D = LaplacianSpectral()
    d = D.dist(G1, G2)
    return d


def NE(G1, G2):
    D = LaplacianSpectral()
    d = D.dist(G1, G2, measure='euclidean')
    return d


all_distance_function_names = ['Jaccard', 'Hamming', 'HammingIpsenMikhailov',
                               'Frobenius', 'PolynomialDissimilarity',
                               'DegreeDivergence', 'PortraitDivergence',
                               'QuantumJSD', 'CommunicabilityJSD',
                               'GraphDiffusion', 'ResistancePerturbation',
                               'NetLSD', 'LaplacianSpectralGJS',
                               'LaplacianSpectralEUL', 'IpsenMikhailov',
                               'NonBacktrackingSpectral', 'DistributionalNBD',
                               'DMeasure', 'DeltaCon', 'NetSimile']

all_distance_functions = [JaccardDistance().dist, Hamming().dist,
                          HammingIpsenMikhailov().dist, Frobenius().dist,
                          PolynomialDissimilarity().dist,
                          DegreeDivergence().dist, PortraitDivergence().dist,
                          QuantumJSD().dist, CommunicabilityJSD().dist,
                          GraphDiffusion().dist, ResistancePerturbation().dist,
                          NetLSD().dist, NJ, LE, IpsenMikhailov().dist,
                          NonBacktrackingSpectral().dist,
                          DistributionalNBD().dist, DMeasure().dist,
                          DeltaCon().dist, NetSimile().dist]

all_distances = dict(zip(all_distance_function_names, all_distance_functions))

df_out = pd.DataFrame(columns=['ensemble', 'n', 'k', 'param_label',
                               'param_value', 'distance_label', 'd'])


ensemble = str(sys.argv[1])
nmax = int(sys.argv[2])
ntimes = int(sys.argv[3])

if ensemble == 'gnp':
    p = 0.1

if ensemble == 'gnk':
    k = 6

gran = 21
param_label = 'n'
param_vals = np.logspace(1.5, np.log10(nmax), gran, dtype=int)

for ni, n in enumerate(param_vals):
    if ni % 4 == 0:
        print(ni)
    for _ in range(ntimes):
        if ensemble == 'gnp':
            G1 = nx.erdos_renyi_graph(n, p)
            G2 = nx.erdos_renyi_graph(n, p)
            k1 = np.mean(list(dict(G1.degree()).values()))
            k2 = np.mean(list(dict(G2.degree()).values()))
            kx = np.mean([k1, k2]).round(5)
        if ensemble == 'gnk':
            G1 = erdos_renyi_graph(n, k)
            G2 = erdos_renyi_graph(n, k)
            kx = k

        for dist_name, distance_i in all_distances.items():
            try:
                d_i = distance_i(G1, G2)
            except:
                d_i = np.nan

            df_i = pd.DataFrame({'ensemble': [ensemble],
                                 'n': [n], 'k': [kx],
                                 'param_label': [param_label],
                                 'param_value': [n],
                                 'distance_label': [dist_name], 'd': [d_i]})
            df_out = pd.concat([df_out, df_i])

df_out = df_out.sort_values('param_value').reset_index().iloc[:, 1:]
s_i = np.random.randint(10000000)

fn = 'out/vary_n_%s_wegd_%s_%06i.csv'%(ensemble, param_label, s_i)
df_out.to_csv(fn, index=False)
