from network_generators import random_geometric_graph
import numpy as np
import networkx as nx
import sys
from netrd.distance import *
from graph_diffusion import GraphDiffusion
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=IntegrationWarning)


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
n = int(sys.argv[2])
ntimes = int(sys.argv[3])
ensemble = 'rgg'

gran = 100
param_label = 'k'
param_vals = np.logspace(np.log(10/(n)), np.log10(n-1), gran).round(7)

for ki, k in enumerate(param_vals):
    if ki % 4 == 0:
        print(ki)
    for _ in range(ntimes):
        G1 = random_geometric_graph(n, k)
        G2 = random_geometric_graph(n, k)

        for dist_name, distance_i in all_distances.items():
            try:
                d_i = distance_i(G1, G2)
            except:
                d_i = np.nan

            df_i = pd.DataFrame({'ensemble': [ensemble],
                                 'n': [n], 'k': [k],
                                 'param_label': [param_label],
                                 'param_value': [k],
                                 'distance_label': [dist_name], 'd': [d_i]})
            df_out = pd.concat([df_out, df_i])

df_out = df_out.sort_values('param_value').reset_index().iloc[:, 1:]
s_i = np.random.randint(1000000)

fn = 'out/%s_wegd_%s_n%i_%06i.csv'%(ensemble, param_label, n, s_i)
df_out.to_csv(fn, index=False)
