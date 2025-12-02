# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


def conflict_graph(graph_c_directed):
    graph_i = nx.Graph()  # Line graph should be undirected
    link_list = list(graph_c_directed.edges())

    # Step 5: Connect nodes in the line graph if they share a common vertex
    for i, (u1, v1) in enumerate(link_list):
        for j, (u2, v2) in enumerate(link_list):
            if i >= j:
                continue  # Avoid duplicate checks
            if (u1, v1) != (u2, v2) and (u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2):
                graph_i.add_edge((u1, v1), (u2, v2))  # Connect the nodes

    return graph_i


def luby_mis_nstep_redraw(adj, wts, nstep=1):
    wts = np.array(wts).flatten()
    verts = np.array(range(wts.size))
    mwis = set()
    remain = set(verts.flatten())
    vidx = list(remain)
    nb_is = set()
    step = nstep

    round_results = []
    round_count = 0

    while len(remain) > 0 and (step or (nstep == -1)):
        round_count += 1
        round_mwis = set()
        pri = wts * np.random.uniform(0.0, 1.0, size=wts.size)

        for v in list(remain):
            _, nb_set = np.nonzero(adj[v])
            nb_set = set(nb_set).intersection(remain)

            if len(nb_set) == 0:
                mwis.add(v)
                round_mwis.add(v)
                continue

            nb_list = sorted(nb_set)
            wts_nb = pri[nb_list]
            w_bar_v = wts_nb.max()

            if pri[v] > w_bar_v:
                mwis.add(v)
                round_mwis.add(v)
                nb_is = nb_is.union(nb_set)
            elif pri[v] == w_bar_v:
                i = list(wts_nb).index(pri[v])
                nbv = nb_list[i]
                if v < nbv:
                    mwis.add(v)
                    round_mwis.add(v)
                    nb_is = nb_is.union(nb_set)
            else:
                pass
        
        round_info = {
            'round': step,
            'selected_vertices': round_mwis,
            'num_selected': len(round_mwis)
        }
        round_results.append(round_info)

        remain = remain - mwis - nb_is
        step -= 1

    # total_ws = np.sum(wts[list(mwis)]) if len(mwis) > 0 else 0.0
    return mwis, round_results, nb_is

def vis_edges(graph, pos, edge_labels, ax=None, font_size = 12):
    nx.draw_networkx_edge_labels(graph, pos = pos, edge_labels = edge_labels, ax =ax, font_size = font_size)

def vis_network(graph, src_nodes, dst_nodes, pos, weights=None, delays=None, with_labels=True, ax=None,
                colors=['g','r','b'], alpha=1.0
                ):
    color_list = ['b', 'm', 'g', 'r', 'k', 'w']
    g_size = graph.number_of_nodes()
    node_colors = ['y' for node in range(g_size)]
    node_sizes = list(np.ones((g_size,)) * 300) # [300 for node in range(g_size)]
    node_shapes = ['o' for node in range(g_size)]
    edge_colors = ['k' for edge in range(len(weights))]
    for i in range(len(weights)):
        if weights[i] > 1.0:
            edge_colors[i] = colors[0]

    if delays is not None:
        # node_sizes = 10 * delays + 5
        node_sizes = (delays/5)**2 + 20

    for i in range(len(src_nodes)):
        node_colors[src_nodes[i]] = colors[1]
        # node_colors[src_nodes[i]] = color_list[i]
        node_shapes[src_nodes[i]] = 'd'
        # if delays is None:
        # if node_sizes[src_nodes[i]] < 200:
        #     node_sizes[src_nodes[i]] = 200

    for i in range(len(dst_nodes)):
        node_colors[dst_nodes[i]] = colors[2]
        # node_colors[dst_nodes[i]] = color_list[i]
        # node_sizes[dst_nodes[i]] = 200
        node_shapes[dst_nodes[i]] = 's'

    nx.draw(
        graph,
        node_color=node_colors,
        node_size=node_sizes,
        with_labels=with_labels,
        pos=pos,
        width=weights,
        ax=ax,
        edge_color=edge_colors,
        alpha=alpha,
        )
    return None

def all_pairs_shortest_paths(graph, weight=None):
    g_size = graph.number_of_nodes()
    sp_mtx = np.zeros((g_size, g_size))
    lengths = nx.all_pairs_dijkstra_path_length(graph, weight=weight)
    lengths = dict(lengths)
    for n1 in graph.nodes:
        for n2 in graph.nodes:
            # sp_mtx[n1][n2] = nx.shortest_path_length(graph, n1, n2)
            sp_mtx[n1][n2] = lengths[n1][n2]
    return sp_mtx


def softmax(x_in, alpha=1.0):
    x = x_in[:]
    ans = np.exp( alpha * x) / sum(np.exp(alpha * x))
    return ans

def power_law_probabilities(x_in, beta=2.0):
    x = x_in[:]
    ans = (x ** beta) / sum(x ** beta)
    return ans

def elu(x_in, alpha=1.0, bias=1.0):
    x = x_in[:]
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1) + bias)

def alpha_sigmoid(x_in, alpha):
    x = x_in[:]
    return 1 / (1 + np.exp(-alpha * x))

def rank_based_probabilities(x_in):
    x = x_in[:]
    ranks = np.argsort(-x) + 1  # 1-based ranks
    ans = 1 / ranks.astype(float)
    ans /= np.sum(ans)
    return ans

def pickle_file_exists(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path) and file_path.endswith('.pkl')


def calculate_cost(hops, delays):
    cost_list = []
    if len(hops) != 0:
        for hop, delay in zip(hops, delays):
            cost = hop ** 1 + delay ** 1
            cost_list.append(cost)
        lowest_cost = min(cost_list)
        lowest_cost_index = cost_list.index(lowest_cost)
    else:
        lowest_cost = None
        lowest_cost_index = None
    return lowest_cost_index, lowest_cost
