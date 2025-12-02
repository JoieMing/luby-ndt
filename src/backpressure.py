# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function

import queue
import sys
import os
import time
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
import scipy.io as sio
import sparse
from copy import deepcopy
np.set_printoptions(threshold=np.inf)
# Import utility functions
from util import *



class BackpressureAnt:
    def __init__(self, num_nodes, T, seed=3, m=2, pos=None, cf_radius=0.0, gtype='ba'):
        self.num_nodes = int(num_nodes)
        self.T = int(T)
        self.t_recordings = [self.T-1]
        self.seed = int(seed) # other format such as int64 won't work
        self.m = int(m)
        self.gtype = gtype.lower()
        self.trace = True
        self.cf_radius = cf_radius
        self.case_name = 'AntBP_seed_{}_nodes_{}_{}'.format(self.seed, self.num_nodes, self.gtype)
        if self.gtype == 'ba':
            graph_c = nx.barabasi_albert_graph(self.num_nodes, self.m, seed=self.seed)  # Conectivity graph
        elif self.gtype == 'grp':
            graph_c = nx.gaussian_random_partition_graph(self.num_nodes, 15, 3, 0.4, 0.2, seed=self.seed)  # Conectivity graph
        elif self.gtype == 'ws':
            graph_c = nx.connected_watts_strogatz_graph(self.num_nodes, k=6, p=0.2, seed=self.seed)  # Conectivity graph
        elif self.gtype == 'er':
            graph_c = nx.fast_gnp_random_graph(self.num_nodes, 15.0/float(self.num_nodes), seed=self.seed)  # Conectivity graph
        elif '.mat' in self.gtype:
            postfix = self.gtype.split('/')[-1]
            postfix = postfix.split('.')[0]
            self.case_name = 'seed_{}_nodes_{}_{}'.format(self.seed, self.num_nodes, postfix)
            try:
                mat_contents = sio.loadmat(self.gtype)
                adj = mat_contents['adj'].todense()
                pos = mat_contents['pos_c']
                graph_c = nx.from_numpy_array(adj)
            except:
                raise RuntimeError("Error creating object, check {}".format(self.gtype))
        elif self.gtype == 'test_a':
            # Test graph A: 0---1---2 (linear path)
            graph_c = nx.path_graph(3)
        elif self.gtype == 'test_b':
            graph_c = nx.path_graph(4)
        elif self.gtype == 'test_c':
            graph_c = nx.Graph()
            graph_c.add_nodes_from([0, 1, 2, 3])
            graph_c.add_edges_from([(0, 1), (1, 2), (1, 3)])  # Node 1 connects to 0,2,3
        else:
            raise ValueError("unsupported graph model for connectivity graph")
        self.connected = nx.is_connected(graph_c)


        self.graph_c = graph_c
        self.node_positions(pos)
        self.box = self.bbox()
        self.graph_i = nx.line_graph(self.graph_c)  # Conflict graph
        self.adj_c = nx.adjacency_matrix(self.graph_c)
        self.num_links = len(self.graph_i.nodes)
        self.num_di_links = 2 * self.num_links
        self.link_list = list(self.graph_i.nodes)
        self.edge_maps = np.zeros((self.num_di_links,), dtype=int)
        self.edge_maps_rev = np.zeros((self.num_di_links,), dtype=int)
        self.link_mapping()
        if cf_radius > 0.5:
            self.add_conflict_relations(cf_radius)
        else:
            self.adj_i = nx.adjacency_matrix(self.graph_i)
        self.mean_conflict_degree = np.mean(self.adj_i.sum(axis=0))
        # self.fid_cmd_map = np.zeros((self.num_nodes,), dtype=int)
        self.fid_cmd_map = np.full((self.num_nodes,), -1, dtype=int)
        self.clear_all_flows()
        self.pheromone_freezed = False
        self.queue_lengths = np.zeros((self.num_nodes, self.num_nodes), dtype=float) # repurpose for ph_routing
        if not self.trace:
            self.delivery = sparse.COO(np.zeros((self.num_nodes, self.num_nodes, self.num_nodes), dtype=float))

        # # Verify path existence
        # if self.gtype == 'test_a':
        #     path = nx.shortest_path(self.graph_c, 0, 2)
        #     print(f"Path 0→2: {path}")
        #     assert path == [0, 1, 2], "Test A path incorrect!"

        # elif self.gtype == 'test_b':
        #     path = nx.shortest_path(self.graph_c, 0, 3)
        #     print(f"Path 0→3: {path}")
        #     assert path == [0, 1, 2, 3], "Test B path incorrect!"

        # elif self.gtype == 'test_c':
        #     path1 = nx.shortest_path(self.graph_c, 0, 2)
        #     path2 = nx.shortest_path(self.graph_c, 0, 3)
        #     print(f"Path 0→2: {path1}")
        #     print(f"Path 0→3: {path2}")
        #     assert path1 == [0, 1, 2], "Test C path 0→2 incorrect!"
        #     assert path2 == [0, 1, 3], "Test C path 0→3 incorrect!"

    def random_walk(self, ss=0.1, n=10):
        disconnected = True
        while disconnected:
            mask = np.random.choice(np.arange(0, self.num_nodes), size=n, replace=False)
            d_pos = np.random.normal(0, ss, size=(n, 2))
            pos_c_np = self.pos_c_np
            pos_c_np[mask, :] += d_pos
            b_min = np.min(self.box)
            b_max = np.max(self.box)
            pos_c_np = pos_c_np.clip(b_min, b_max)
            d_mtx = distance_matrix(pos_c_np, pos_c_np)
            adj_mtx = np.zeros([self.num_nodes, self.num_nodes], dtype=int)
            adj_mtx[d_mtx <= 1.0] = 1
            np.fill_diagonal(adj_mtx, 0)
            graph_c = nx.from_numpy_array(adj_mtx)
            self.connected = nx.is_connected(graph_c)
            disconnected = not self.connected
        return graph_c, pos_c_np


    class Flow:
        def __init__(self, source_node, arrival_rate, dest_node):
            self.source_node = source_node
            self.arrival_rate = arrival_rate
            self.dest_node = dest_node
            self.cut_off = -1

    def node_positions(self, pos):
        if pos is None:
            # ✅ Provide predefined positions for test graphs
            if self.gtype in ['test_a', 'test_b', 'test_c']:
                if self.gtype == 'test_a':
                    # Test A: 0---1---2 (linear layout)
                    pos_c = {0: (0.0, 0.5), 1: (0.5, 0.5), 2: (1.0, 0.5)}
                elif self.gtype == 'test_b':
                    # Test B: 0---1---2---3 (linear layout)
                    pos_c = {0: (0.0, 0.5), 1: (0.33, 0.5), 2: (0.66, 0.5), 3: (1.0, 0.5)}
                elif self.gtype == 'test_c':
                    # Test C: 0---1---2, 1---3 (star layout, node 1 as center)
                    pos_c = {
                        0: (0.0, 0.5),    # left
                        1: (0.5, 0.5),    # center
                        2: (1.0, 0.5),    # right  
                        3: (0.5, 0.0)     # bottom
                    }
            else:
                pos_file = os.path.join("..", "pos", "graph_c_pos_{}.p".format(self.case_name))
                if not os.path.isfile(pos_file):
                    pos_c = nx.spring_layout(self.graph_c)
                    with open(pos_file, 'wb') as fp:
                        pickle.dump(pos_c, fp, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(pos_file, 'rb') as fp:
                        pos_c = pickle.load(fp)
        elif isinstance(pos, str) and pos == 'new':
            pos_c = nx.spring_layout(self.graph_c)
        elif isinstance(pos, np.ndarray):
            pos_c = dict(zip(list(range(self.num_nodes)), pos))
        else:
            raise ValueError("unsupported pos format in backpressure object initialization")
        self.pos_c = pos_c

    def bbox(self):
        pos_c = np.zeros((self.num_nodes, 2))
        for i in range(self.num_nodes):
            pos_c[i, :] = self.pos_c[i]
        self.pos_c_np = pos_c
        return [np.amin(pos_c[:,0])-0.05, np.amax(pos_c[:,1])+0.05, np.amin(pos_c[:,1])-0.12, np.amax(pos_c[:,1])+0.05]

    def add_conflict_relations(self, cf_radius):
        """
        Adding conflict relationship between links whose nodes are within cf_radius * median_link_distance
        :param cf_radius: multiple of median link distance
        :return: None (modify self.adj_i, and self.graph_i inplace)
        """
        pos_c_vec = np.zeros((self.num_nodes, 2))
        for key, item in self.pos_c.items():
            pos_c_vec[key, :] = item
        dist_mtx = distance_matrix(pos_c_vec, pos_c_vec)
        rows, cols = np.nonzero(self.adj_c)
        link_dist = dist_mtx[rows, cols]
        median_dist = np.nanmedian(link_dist)
        intf_dist = cf_radius * median_dist
        for link in self.link_list:
            src, dst = link
            intf_nbs_s, = np.where(dist_mtx[src, :] < intf_dist)
            intf_nbs_d, = np.where(dist_mtx[dst, :] < intf_dist)
            intf_nbs = np.union1d(intf_nbs_s, intf_nbs_d)
            for v in intf_nbs:
                _, nb2hop = np.nonzero(self.adj_c[v])
                for u in nb2hop:
                    if {v, u} == {src, dst}:
                        continue
                    elif (v, u) in self.link_list:
                        self.graph_i.add_edge((v, u), (src, dst))
                    elif (u, v) in self.link_list:
                        self.graph_i.add_edge((u, v), (src, dst))
                    else:
                        pass
                        # raise RuntimeError("Something wrong with adding conflicting edge")
        self.adj_i = nx.adjacency_matrix(self.graph_i)

    def link_mapping(self):
        # Mapping between links in connectivity graph and nodes in conflict graph
        j = 0
        for e0, e1 in self.graph_c.edges:
            try:
                i = self.link_list.index((e0, e1))
            except:
                i = self.link_list.index((e1, e0))
            # Link direction A
            self.edge_maps[j] = i
            self.edge_maps_rev[i] = j
            # Link direction B
            self.edge_maps[j + self.num_links] = i + self.num_links
            self.edge_maps_rev[i + self.num_links] = j + self.num_links
            j += 1

    def add_flow(self, src, dst, rate=2.0, start=0, cutoff=-1):
        # used for icassp submission
        fi = self.Flow(src, rate, dst)
        fi.start_time = self.T if start > self.T else max(start, 0)
        if 0 < cutoff < self.T:
            fi.cut_off = int(cutoff)
        else:
            fi.cut_off = self.T
        self.flows.append(fi)
        self.src_nodes.append(src)
        self.dst_nodes.append(dst)
        self.num_flows = len(self.flows)

    def clear_all_flows(self):
        self.flows = []
        self.num_flows = 0
        self.src_nodes = []
        self.dst_nodes = []
        # self.fid_cmd_map.fill(np.nan)
        self.fid_cmd_map.fill(-1)  # Use -1 to indicate unmapped

    def flows_init(self):
        self.flows_sink_departures = np.zeros((self.num_flows, self.T), dtype=int)
        self.flows_arrivals = np.zeros((self.num_flows, self.T), dtype=int)
        self.flow_pkts_in_network = np.zeros((self.num_flows, self.T), dtype=int)
        np.random.seed(self.seed)
        self.fid_cmd_map.fill(-1)
        for fidx in range(self.num_flows):
            arrival_rate = self.flows[fidx].arrival_rate
            T = int(self.flows[fidx].cut_off)
            self.flows_arrivals[fidx, 0:T] = np.random.poisson(arrival_rate, size=(T,))
            dst = self.flows[fidx].dest_node
            if self.fid_cmd_map[dst] == -1:    # Set only once
                self.fid_cmd_map[dst] = fidx
            # self.fid_cmd_map[self.flows[fidx].dest_node] = fidx

    def flows_reset(self):
        self.flows_sink_departures = np.zeros((self.num_flows, self.T), dtype=int)
        self.flow_pkts_in_network = np.zeros((self.num_flows, self.T), dtype=int)

    def freeze_pherom(self):
        self.pheromone_freezed = True

    def unfreeze_pherom(self):
        self.pheromone_freezed = False

    def links_init(self, rates, std=2):
        if hasattr(rates, '__len__'):
            assert len(rates) == self.num_links
            stds = std * np.ones_like(rates)
        else:
            stds = std
        link_rates = np.zeros((self.num_links, self.T))
        for t in range(self.T):
            link_rates[:, t] = np.clip(np.random.normal(rates, stds), 0, rates + 3 * std)
        self.link_rates = np.round(link_rates)
        # print("link rates initialized:", self.link_rates)

    # def link_failure(self, link_bias):
    #     deleted_link = np.random.choice(self.link_list)
    #     deleted_link_index = self.link_list.index(deleted_link)
    #     self.link_list.remove(deleted_link)
    #     self.pheromones = np.delete(self.pheromones, deleted_link_index+self.num_links, axis=0)
    #     self.pheromones = np.delete(self.pheromones, deleted_link_index, axis=0)
    #     self.pheromones_vis = np.delete(self.pheromones_vis, deleted_link_index+self.num_links, axis=0)
    #     self.pheromones_vis = np.delete(self.pheromones_vis, deleted_link_index, axis=0)
    #     link_bias = np.delete(link_bias, deleted_link_index, axis=0)
    #     return link_bias

    def queues_init(self):
        # Initialize system state
        self.queue_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self.W = np.zeros((self.num_links, self.T))
        self.WSign = np.ones((self.num_links, self.T))
        self.opt_comd_mtx = -np.ones((self.num_links, self.T), dtype=int)
        self.link_comd_cnts = np.zeros((self.num_links, self.num_flows))
        self.di_link_comd_cnts = np.zeros((self.num_di_links, self.num_flows))
        self.pkt_vis = np.zeros((self.num_di_links, self.num_flows, self.T))
        self.backlog = {}
        self.backlog_ph = {}
        for i in range(self.num_nodes):
            backlog_i = {}
            backlog_i_ph = {}
            for j in range(self.num_nodes + 1):
                # Each queue holds pkt from node i to node j
                # queue backlog_ph[i][i] holds undetermined pkts
                # queue backlog_ph[i][self.num_nodes] holds pkt reach its destination node i
                qi = queue.Queue()
                backlog_i[j] = qi
                qi_ph = queue.Queue()
                backlog_i_ph[j] = qi_ph
            self.backlog[i] = backlog_i
            self.backlog_ph[i] = backlog_i_ph
        self.queue_lengths = np.zeros((self.num_nodes, self.num_nodes))
        self.HOL_t0 = np.zeros((self.num_nodes, self.num_nodes))
        self.HOL_delay = np.zeros((self.num_nodes, self.num_nodes))
        self.SJT_delay = np.zeros((self.num_nodes, self.num_nodes))

        self.sched_count = np.zeros((self.num_links,), dtype=int)      # undirected links


    # def pheromone_init(self, decay=0.97, unit=0.01, init=0.5):
    #     self.phmns_decay = decay
    #     self.phmns_unit = unit
    #     # The shape of pheromones is total number of directional links x total number of destinations
    #     link_rate_avg = np.nanmean(self.link_rates)
    #     if decay < 1.0:
    #         max_val = link_rate_avg * unit / (1.0 - decay)
    #     else:
    #         max_val = 1.0
    #     self.pheromones = init * max_val * np.ones((self.num_di_links, self.num_flows), dtype=float)
    #     self.pheromones_vis = np.zeros((self.num_di_links, self.num_flows, 1), dtype=float)
    #     self.queue_matrix_exp = np.zeros_like(self.queue_matrix)
    #     self.phmns_exp = 1 + (1 - decay)
    def pheromone_init(self, decay=0.97, unit=0.01, init=0.5):
        self.phmns_decay = decay
        self.phmns_unit = unit
        # The shape of pheromones is total number of directional links x total number of destinations
        self.pheromones = np.zeros((self.num_di_links, self.num_flows), dtype=float)
        self.pheromones_vis = np.zeros((self.num_di_links, self.num_flows, 1), dtype=float)
        self.queue_matrix_exp = np.zeros_like(self.queue_matrix)
        self.phmns_exp = 1 + (1 - decay)

    def computing_init(self, clients, servers, proc_bws, service_caps, delay_est):
        self.clients = clients
        self.servers = servers
        self.proc_bws = proc_bws
        self.service_capacity = service_caps
        graph_ext = deepcopy(self.graph_c)
        avg_delay = np.mean(delay_est)
        scale = avg_delay**2
        for link, delay in zip(self.link_list, delay_est):
            src, dst = link
            graph_ext[src][dst]["delay"] = delay

        virtual_sink = self.num_nodes
        graph_ext.add_node(virtual_sink)
        for n in range(self.num_nodes):
            if service_caps[n] > 0:
                vlink_wt = scale/service_caps[n]
                graph_ext.add_edge(n, virtual_sink, delay=vlink_wt)
            else:
                self.service_capacity[n] = np.nan
        return graph_ext

    def bias_diff(self, bias_matrix):
        link_bias = np.zeros((self.num_links, self.num_nodes), dtype=float)
        for lidx in range(self.num_links):
            src, dst = self.link_list[lidx]
            bdiff = bias_matrix[src, :] - bias_matrix[dst, :]
            link_bias[lidx, :] = bdiff
        return link_bias

    def pkt_arrival(self, t):
        for fidx in range(self.num_flows):
            flow = self.flows[fidx]
            src = flow.source_node
            dst = flow.dest_node
            self.queue_matrix[src, dst] += self.flows_arrivals[fidx, t]
            self.queue_lengths[src, src] += self.flows_arrivals[fidx, t]
            self.queue_matrix_exp[src, dst] += self.flows_arrivals[fidx, t]
            for i in range(self.flows_arrivals[fidx, t]):
                self.backlog[src][dst].put((t, t))
                # pheromone routing, queue to itself means undetermined
                self.backlog_ph[src][src].put((t, t, dst, None))

    def update_HOL_matrix(self, t):
        '''should be run after packet arrivals'''
        if self.trace:
            for src in range(self.num_nodes):
                for cmd in self.dst_nodes:
                    if self.backlog[src][cmd].empty() or (src == cmd):
                        self.HOL_delay[src][cmd] = 0
                    else:
                        pkt = self.backlog[src][cmd].queue[0]
                        t0, t1 = pkt
                        self.HOL_t0[src][cmd] = t0
                        self.HOL_delay[src][cmd] = t - t1

    def update_SJT_matrix(self, t):
        '''should be run after packet arrivals'''
        if self.trace:
            for src in range(self.num_nodes):
                for cmd in self.dst_nodes:
                    self.SJT_delay[src][cmd] = 0
                    if self.backlog[src][cmd].empty() or (src == cmd):
                        pass
                    else:
                        for pkt in self.backlog[src][cmd].queue:
                            t0, t1 = pkt
                            self.SJT_delay[src][cmd] += t - t1
    
    def commodity_selection(self, queue_mtx, mbp=0.0, link_phmn=None):
        W_amp = np.zeros((self.num_links,), dtype=float)
        W_sign = np.ones((self.num_links,), dtype=float)
        comds = -np.ones((self.num_links,), dtype=int)
        j = 0
        for link in self.link_list:
            wts_link = queue_mtx[link[0], self.dst_nodes] - queue_mtx[link[1], self.dst_nodes]
            directions = np.sign(wts_link)
            # find out the source nodes
            ql_src_vec = np.where(directions > 0.0,
                                  self.queue_matrix[link[0], self.dst_nodes],
                                  self.queue_matrix[link[1], self.dst_nodes])
            # create a mask that source nodes has more than 1 packet to transmit
            ql_mask = np.where(ql_src_vec > 0.1, np.ones_like(self.dst_nodes), np.zeros_like(self.dst_nodes))
            if link_phmn is None:
                wts_link = np.multiply(wts_link, ql_mask)
            else:
                wts_link = np.multiply(wts_link + link_phmn[j, self.dst_nodes], ql_mask)
            cmd = np.argmax(abs(wts_link))
            W_sign[j] = np.sign(wts_link[cmd])
            W_amp[j] = np.amax([abs(wts_link[cmd]) - mbp, 0])
            comds[j] = self.dst_nodes[cmd] if np.amax(abs(wts_link)) > 0.0 else -1
            # if W_sign[j] == 1:
            #     ql_src = self.queue_matrix[link[0], self.dst_nodes[cmd]]
            # else:
            #     ql_src = self.queue_matrix[link[1], self.dst_nodes[cmd]]
            # comds[j] = self.dst_nodes[cmd] if (np.amax(abs(wts_link)) > 0 and ql_src > 0) else -1
            j += 1
        return W_amp, W_sign, comds

    def ph_routing(self, t, func='proportional', exploration_rate=0.0, not_going_back=False, link_bias=None, ph_diff=False):
        W_amp = np.zeros((self.num_links,), dtype=float)
        W_sign = np.ones((self.num_links,), dtype=float)
        for v in range(self.num_nodes):
            n_undecided = self.backlog_ph[v][v].qsize()
            if n_undecided == 0:
                continue
            _, nb_set = np.nonzero(self.adj_c[v])
            rlinks = -np.ones_like(nb_set)
            dlinks = -np.ones_like(nb_set)
            for j in range(len(nb_set)):
                u = nb_set[j]
                if (v, u) in self.link_list:
                    i = self.link_list.index((v, u))
                    rlinks[j] = i + self.num_links
                    dlinks[j] = i
                elif (u, v) in self.link_list:
                    i = self.link_list.index((u, v))
                    rlinks[j] = i
                    dlinks[j] = i + self.num_links
                else:
                    pass
                #dlinks[j] = i
            for j in range(n_undecided):
                pkt = self.backlog_ph[v][v].get_nowait()
                if pkt is None:
                    raise RuntimeError("Ph_routing Backlog error node: {}".format(v))
                t0, t1, cmd, last = pkt
                cmd_fid = self.fid_cmd_map[cmd]
                # Drop packets trapped in the network for too long
                # if t1 - t0 > 100:
                #     continue
                if last is not None and not_going_back:
                    # do something here to nb_set to avoid pkt being sent back to its last node
                    remov = np.where(nb_set == last)[0]
                    dlinks = np.delete(dlinks, remov)
                    rlinks = np.delete(rlinks, remov)
                    nb_set = np.delete(nb_set, remov)
                # select outbound node
                if nb_set.size < 1.0:
                    continue

                if ph_diff:
                    phs = self.pheromones[dlinks, cmd_fid] - self.pheromones[rlinks, cmd_fid]
                else:
                    phs = self.pheromones[dlinks, cmd_fid]

                if link_bias is not None:
                    dilink_bias = np.vstack((link_bias, -link_bias))
                    phs += dilink_bias[dlinks, cmd]

                rand_value = np.random.rand()
                if rand_value < exploration_rate:
                    probs = np.ones_like(nb_set)/float(nb_set.size)
                else:
                    if func == 'proportional':
                        phs[phs < 0] = 0
                        if np.all(phs == 0):
                            probs = np.ones_like(nb_set) / float(nb_set.size)
                        else:
                            probs = phs / np.sum(phs)
                    elif func == 'softmax':
                        probs = softmax(phs, alpha=1)
                    elif func == 'powerlaw':
                        probs = power_law_probabilities(phs, beta=2)
                    elif func == 'rankbased':
                        probs = rank_based_probabilities(phs)
                    elif func == 'elu':
                        probs = elu(phs)
                        if np.all(probs == 0):
                            probs = np.ones_like(nb_set) / float(nb_set.size)
                        else:
                            probs = probs / np.sum(probs)
                has_nan_values = np.isnan(probs).any()
                if has_nan_values:
                    continue
                u = np.random.choice(nb_set, p=probs)
                self.backlog_ph[v][u].put((t0, t, cmd, v))
                self.queue_lengths[v, u] += 1
                self.queue_lengths[v, v] -= 1

        # routing is finished up to here, but we also update the following values to be compatible with BP routing
        for j in range(self.num_links):
            v, u = self.link_list[j]
            wts_link = max(self.queue_lengths[v, u], self.queue_lengths[u, v])
            W_sign[j] = np.sign(self.queue_lengths[v, u] - self.queue_lengths[u, v] + 0.01)
            # create a mask that source nodes has more than 1 packet to transmit
            ql_mask = 1 if wts_link > 0.1 else 0
            wts_link = np.multiply(wts_link, ql_mask)
            W_amp[j] = float(wts_link)

        return W_amp, W_sign

    def estimate_link_flow_rates(self, weight='delay'):
        """
        Estimate the sum rate of all flows on each link for SP-only routing.
        Since routing is deterministic (shortest path), we can calculate this analytically.
        
        Returns:
            link_flow_rates: (num_links,) array of total flow rates on each undirected link
        """
        link_flow_rates = np.zeros(self.num_links, dtype=float)

        for fidx, flow in enumerate(self.flows):
            src = flow.source_node
            dst = flow.dest_node
            flow_rate = flow.arrival_rate
            
            try:
                # Get shortest path for this flow
                path = nx.shortest_path(self.graph_c, source=src, target=dst, weight=weight)
            except nx.NetworkXNoPath:
                continue
                
            if len(path) < 2:
                continue
                
            # Add flow rate to each link in the path
            for u, v in zip(path[:-1], path[1:]):
                if u >= self.num_nodes or v >= self.num_nodes:
                    continue
                try:
                    # Find the directed link index
                    i = self.link_list.index((u, v))
                except ValueError:
                    # Link is in reverse direction
                    i = self.link_list.index((v, u))

                link_flow_rates[i] += flow_rate
                
        return link_flow_rates

    def estimate_link_busylevel(self, weight='delay', link_rates=None):
        """
        Estimate link utilization = sum_flow_rates / link_capacity for each link.
        IMPORTANT: Ensures stability by clipping busy levels to [0, 1] to guarantee λ ≤ μ
        
        Returns:
            utilization: (num_links,) array of utilization for each undirected link, clipped to [0, 1]
        """
        link_flow_rates = self.estimate_link_flow_rates(weight)
        busy_level = np.zeros(self.num_links, dtype=float)
        
        # Track overloaded links for debugging
        overloaded_links = []
        
        for i in range(self.num_links):
            # For undirected links, take max of both directions
            sum_rate = link_flow_rates[i]
            
            # Average link capacity over time
            avg_capacity = link_rates[i]
            if avg_capacity > 0:
                # busy_level[i] = sum_rate / avg_capacity
                raw_busy_level = sum_rate / avg_capacity
                
                # Check for overload (λ > μ)
                if raw_busy_level > 1.0:
                    overloaded_links.append((i, self.link_list[i], sum_rate, avg_capacity, raw_busy_level))
                
                # Clip to ensure stability (λ ≤ μ)
                busy_level[i] = min(raw_busy_level, 1.0)
        
        # Warning for overloaded links
        if overloaded_links:
            print(f"WARNING: {len(overloaded_links)} overloaded links detected (λ > μ):")
            for link_idx, link, flow_rate, capacity, ratio in overloaded_links:
                print(f"  Link {link_idx} {link}: flow_rate={flow_rate:.3f}, capacity={capacity:.3f}, ratio={ratio:.3f}")
            print("  Busy levels clipped to 1.0 for stability.")

        return busy_level
    
    def compute_analytical_duty_cycle_v3redraw(self, b_init, b_joint_1, M=5, L=100, eps=1e-12):
        """
        Multi-round DT with *redrawing priorities each round*.

        Parameters
        ----------
        b_init : (E,) array-like
            Initial busy probs b^{(1)} = min(lambda/mu, 1).
        M : int
            Number of contention rounds.
        L : int
            Grid size per link for numerical integral.
        eps : float
            Small constant for numerical safety.
        b_joint_1 : (E,E) array-like or None
            Joint busy probabilities at round-1: b_joint_1[i,e] = b^{(1)}_{i,e}.
            Used to compute conditional b^{(1)}_{i|e} = b^{(1)}_{i,e} / b^{(1)}_e.
            If None, falls back to independence: b^{(1)}_{i|e} = b^{(1)}_i.

        Returns
        -------
        x : (E,)
            Duty cycle accumulated over M rounds: x_e = sum_m b_e^{(m)} P_win_e^{(m)}.
        P_win_round : (M, E)
            Conditional win contributions per round: b^{(m)} * P_win^{(m)}.
        b_round : (M+1, E)
            Busy/entry probabilities per round (b^{(1)}..b^{(M+1)}).
        """
        b_init = np.asarray(b_init, dtype=float).flatten()
        E = b_init.size

        # Priority supports z_e (defaults to 1 if not provided)
        if hasattr(self, "z") and self.z is not None:
            z = np.asarray(self.z, dtype=float).flatten()
            if z.size != E:
                raise ValueError("self.z must have length E")
            z = np.maximum(z, eps)
        else:
            z = np.ones(E, dtype=float)

        # Neighbor lists
        Nbrs = [ self.adj_i[e].nonzero()[1].astype(int) for e in range(E) ]

        # Precompute each link's grid: x_e[l] = l * z_e / L
        L = int(max(1, int(L)))
        x_grid = [ (z[e] * np.arange(L) / float(L)) for e in range(E) ]

        # Storage
        x_accum = np.zeros(E, dtype=float)           # duty cycle
        P_win_round = np.zeros((M, E), dtype=float)  # store b^{(m)} * P_win^{(m)}
        b_round = np.zeros((M+1, E), dtype=float)    # b^{(m)}
        b_round[0, :] = np.clip(b_init, 0.0, 1.0)

        # -------------------- [MOD #1] build conditional busy b_{i|e}^{(1)} --------------------
        # If joint probs provided: b_{i|e}^{(1)} = b_joint_1[i,e] / b_init[e]; else independence.
        if b_joint_1 is not None:
            b_joint_1 = np.asarray(b_joint_1, dtype=float)
            if b_joint_1.shape != (E, E):
                raise ValueError("b_joint_1 must be shape (E, E)")
            # Avoid divide-by-zero; if b_e^{(1)} == 0, set conditionals to 0 except self which is 1.
            with np.errstate(divide='ignore', invalid='ignore'):
                Bcond = b_joint_1 / np.maximum(b_init[None, :], eps)  # (i,e)
            Bcond = np.clip(Bcond, 0.0, 1.0)
        else:
            # independence: b_{i|e}^{(1)} = b^{(1)}_i
            Bcond = np.repeat(b_init[:, None], E, axis=1)

        # For consistency: when conditioning on e itself, b_{e|e}^{(1)} = 1
        np.fill_diagonal(Bcond, 1.0)
        # ----------------------------------------------------------------------------------------

        # Helper for F_{i|e}^{(m)}(x) on absolute x (vectorized over x on e's grid)
        # Using the user's formula: F^{(m)}_{i|e}(x) = (1 - b^{(1)}_{i|e}) + b^{(1)}_{i|e} * clip(x/z_i, 0, 1)
        def Fi_cond_vals_for_abs_x(i, x_abs, b_i_cond):
            t = np.clip(x_abs / z[i], 0.0, 1.0)
            return (1.0 - b_i_cond) + b_i_cond * t

        for m in range(1, M+1):
            b_m = b_round[m-1, :].copy()

            # ---------------- [MOD #2] Per-round win probability with conditional CDFs -----------
            # P^{(m)}_{e,win} ≈ (1/L) * sum_{l=0}^{L-1} prod_{i in N(e)} F^{(m)}_{i|e}(l*z_e/L)
            P_win_m = np.zeros(E, dtype=float)

            for e in range(E):
                Nb = Nbrs[e]
                if Nb.size == 0:
                    P_win_m[e] = 1.0
                    continue

                # Use b^{(1)}_{i|e} per the provided math (conditioning on \hat b_e^{(1)}=1)
                b_cond_e = Bcond[:, e]  # length E; we'll index neighbors

                x_e = x_grid[e]  # length-L in [0, z_e)
                log_prod = np.zeros(L, dtype=float)

                for i in Nb:
                    Fi = Fi_cond_vals_for_abs_x(i, x_e, b_cond_e[i])
                    Fi = np.clip(Fi, eps, 1.0)        # numerical safety
                    log_prod += np.log(Fi)

                # Riemann sum of the integral on [0, z_e]: (1/L) * sum exp(sum log Fi)
                P_win_m[e] = float(np.mean(np.exp(log_prod)))

            P_win_m = np.clip(P_win_m, 0.0, 1.0)
            # store b^{(m)} * P_win^{(m)} (same convention as your original)
            P_win_round[m-1, :] = np.clip(b_m * P_win_m, 0.0, 1.0)
            # -------------------------------------------------------------------------------------

            # ---- Accumulate duty x += b^{(m)} * P_win^{(m)} ----
            x_accum += b_m * P_win_m

            if m == M:
                break

            # ---- Mean-field survival update for b^{(m+1)} (unchanged) ----
            blockers = 1.0 - (b_m * P_win_m)
            b_next = b_m * (1.0 - P_win_m)
            for e in range(E):
                for i in Nbrs[e]:
                    b_next[e] *= blockers[i]
            b_round[m, :] = np.clip(b_next, 0.0, 1.0)

        return x_accum, P_win_round, b_round

    def scheduling_rounds_redraw(self, utility, weights=None, n_rounds=5, seed=None):
        # rng = np.random.default_rng(seed)

        if weights is None:
            weights = np.ones_like(utility)

        keep_index = np.flatnonzero(utility > 0.0)           # Only let active links participate
        if keep_index.size == 0:
            return [], []

        adj = self.adj_i[keep_index, :][:, keep_index]       # Conflict graph submatrix

        mwis, round_results, _ = luby_mis_nstep_redraw(adj, weights[keep_index], nstep=n_rounds)

        # Map each round's subgraph indices back to global link indices
        round_solutions = []
        for round_info in round_results:
            round_subgraph_vertices = round_info['selected_vertices']
            round_global_links = keep_index[np.fromiter(round_subgraph_vertices, dtype=int)].tolist()
            
            round_solutions.append({
                'round': round_info['round'],
                'global_links': round_global_links,
                'num_selected': len(round_global_links)
            })
            
        final_solu = keep_index[np.fromiter(mwis, dtype=int)].tolist()
        return final_solu, round_solutions

    def transmission_ph(self, t, mwis):
        """
        Matrix formed transmission with ph_routing, It takes 0.597 seconds to run 100 time slots on graph (15) seed 3
        :param t: time
        :param mwis: list of scheduled links
        :return:
        """
        dsts = -np.ones((len(mwis),), dtype=int)
        srcs = -np.ones_like(dsts)
        schs = -np.ones_like(dsts)
        schs_di = -np.ones((len(mwis),), dtype=int)
        for idx in range(len(mwis)):
            link = mwis[idx]
            shift = 0
            if self.WSign[link, t] < 0:
                dsts[idx] = self.link_list[link][0]
                srcs[idx] = self.link_list[link][1]
                shift = self.num_links
            elif self.WSign[link, t] > 0:
                srcs[idx] = self.link_list[link][0]
                dsts[idx] = self.link_list[link][1]
            else:
                continue
            schs[idx] = link
            schs_di[idx] = link + shift
        schs_di = schs_di[schs_di != -1]
        schs = schs[schs != -1]
        dsts = dsts[dsts != -1]
        srcs = srcs[srcs != -1]
        num_pkts = np.minimum(self.queue_lengths[srcs, dsts], self.link_rates[schs, t])
        if not self.pheromone_freezed:
            self.pheromones = self.pheromones * self.phmns_decay
        for idx in range(schs.size):
            link = schs[idx]
            dlink = schs_di[idx]
            src = srcs[idx]
            dst = dsts[idx]
            num = num_pkts[idx]
            # print(f'we have link {self.link_list[link]} with source {src} and destination {dst}')
            # print(f'the queue length is {self.queue_lengths[src,dst]} and we want to send {num} packets')
            self.queue_lengths[src, dst] -= num
            for i in range(int(num)):
                pkt = self.backlog_ph[src][dst].get_nowait()
                if pkt is None:
                    raise RuntimeError("Ph_transmission: Backlog error node: {}".format(src))
                t0, t1, cmd, last = pkt
                cmd_fid = self.fid_cmd_map[cmd]
                if not self.pheromone_freezed:
                    self.pheromones[dlink, cmd_fid] += self.phmns_unit
                if dst == cmd:
                    #print(f'commodity is the same as destination')
                    fidx = self.dst_nodes.index(cmd)
                    self.flows_sink_departures[fidx, t] += 1
                    self.backlog_ph[dst][self.num_nodes].put((t0, t, cmd, src))
                else:
                    self.backlog_ph[dst][dst].put((t0, t, cmd, src))
                    self.queue_lengths[dst, dst] += 1
                self.link_comd_cnts[link, cmd_fid] += 1
                self.di_link_comd_cnts[dlink, cmd_fid] += 1
                self.pkt_vis[dlink, cmd_fid, t] += 1
        if t in self.t_recordings:
            self.pheromones_vis = np.concatenate((self.pheromones_vis, self.pheromones[..., np.newaxis]), axis=-1)

    def transmission(self, t, mwis):
        """
        Matrix formed transmission, It takes 0.597 seconds to run 100 time slots on graph (15) seed 3
        :param t: time
        :param mwis: list of scheduled links
        :return:
        """

        dsts = -np.ones((len(mwis),), dtype=int)
        srcs = -np.ones_like(dsts)
        schs = -np.ones_like(dsts)
        schs_di = -np.ones((len(mwis),), dtype=int)
        for idx in range(len(mwis)):
            link = mwis[idx]
            shift = 0
            if self.WSign[link, t] < 0:
                dsts[idx] = self.link_list[link][0]
                srcs[idx] = self.link_list[link][1]
                shift = self.num_links
            elif self.WSign[link, t] > 0:
                srcs[idx] = self.link_list[link][0]
                dsts[idx] = self.link_list[link][1]
            else:
                continue
            schs[idx] = link
            schs_di[idx] = link + shift
        schs_di = schs_di[schs_di != -1]
        schs = schs[schs != -1]
        dsts = dsts[dsts != -1]
        srcs = srcs[srcs != -1]
        opt_comds = self.opt_comd_mtx[schs, t]
        num_pkts = np.minimum(self.queue_matrix[srcs, opt_comds], self.link_rates[schs, t])
        opt_comds_fids = self.fid_cmd_map[opt_comds]
        if self.trace:
            for idx in range(len(mwis)):
                src = srcs[idx]
                dst = dsts[idx]
                num = num_pkts[idx]
                cmd = opt_comds[idx]
                if cmd == -1:
                    continue
                elif dst == cmd:
                    fidx = self.dst_nodes.index(cmd)
                    self.flows_sink_departures[fidx, t] = num
                for i in range(int(num)):
                    pkt = self.backlog[src][cmd].get_nowait()
                    if pkt is None:
                        raise RuntimeError("Backlog error node: {}, commodity: {}".format(src, cmd))
                    t0, t1 = pkt
                    self.backlog[dst][cmd].put((t0, t))
        self.queue_matrix_exp = self.queue_matrix_exp * self.phmns_exp
        queue_exp_per_pkt = np.nan_to_num(
            np.divide(self.queue_matrix_exp[srcs, opt_comds], self.queue_matrix[srcs, opt_comds]), nan=0.0)
        self.queue_matrix_exp[dsts, opt_comds] += num_pkts
        self.queue_matrix_exp[srcs, opt_comds] -= queue_exp_per_pkt * num_pkts
        self.queue_matrix_exp[self.queue_matrix_exp < 0.5] = 0.0
        self.queue_matrix[dsts, opt_comds] += num_pkts
        self.queue_matrix[srcs, opt_comds] -= num_pkts

        if not self.trace:
            coords = np.vstack([srcs, dsts, opt_comds])
            coo_pkts = sparse.COO(coords=coords, data=num_pkts, shape=(self.num_nodes, self.num_nodes, self.num_nodes))
            self.delivery += coo_pkts

        self.link_comd_cnts[schs, opt_comds_fids] += num_pkts
        self.di_link_comd_cnts[schs_di, opt_comds_fids] += num_pkts

        self.pheromones = self.pheromones * self.phmns_decay
        self.pheromones[schs, opt_comds_fids] += self.phmns_unit * np.multiply(num_pkts, self.WSign[mwis, t])
        # there are shadow commodities in SP-bias (0 packets to transmit)
        sink_true = np.logical_and(opt_comds == dsts, num_pkts > 0.1)
        sink_dsts = dsts[sink_true]
        if t in self.t_recordings:
            self.pheromones_vis = np.concatenate((self.pheromones_vis, self.di_link_comd_cnts[..., np.newaxis]), axis=-1)
        if len(sink_dsts) > 0:
            self.queue_matrix[sink_dsts, sink_dsts] = 0
            self.queue_matrix_exp[sink_dsts, sink_dsts] = 0
            if not self.trace:
                fidxs = np.zeros_like(sink_dsts)
                for sidx in range(len(sink_dsts)):
                    fidx, = np.where(self.dst_nodes == sink_dsts[sidx])
                    if len(fidx) > 0:
                        fidxs[sidx] = fidx[0]
                self.flows_sink_departures[fidxs, t] = num_pkts[sink_true]

    def update_bias_mean(self, bias_matrix):
        # step 1: find out neighbors, construct an out adj matrix
        out_adj = np.zeros((self.num_nodes, self.num_nodes))
        bias_matrix_new = np.copy(bias_matrix)
        for cmd in self.dst_nodes:
            for idx_link in range(self.num_links):
                e0, e1 = self.link_list[idx_link]
                val = self.pheromones[idx_link, self.fid_cmd_map[cmd]]
                if val > 0:
                    out_adj[e0, e1] = abs(val)
                elif val < 0:
                    out_adj[e1, e0] = abs(val)
                else:
                    pass
            out_adj = out_adj / np.linalg.norm(out_adj, ord=1, axis=1, keepdims=True)
            # step 2: update bias
            tmp = np.dot(out_adj, bias_matrix[cmd]+1)
            bias_matrix_new[~np.isnan(tmp), cmd] = tmp[~np.isnan(tmp)]
            bias_matrix_new[cmd, cmd] = 0
        return bias_matrix_new

    def update_bias(self, bias_matrix, delay_mtx):
        # step 1: find out neighbors, construct an out adj matrix
        bias_matrix_new = np.copy(bias_matrix)
        for v in range(self.num_nodes):
            _, nb_set = np.nonzero(self.adj_c[v])
            sp_v = (bias_matrix[nb_set, :] + delay_mtx[nb_set, v:v + 1]).min(axis=0)
            bias_matrix_new[v, :] = np.minimum(sp_v, bias_matrix[v, :])
        return bias_matrix_new

    def collect_delay(self, opt, T=1000):
        """
        aggregate based on the servers(dst)
        """
        dsts_all = np.array(self.dst_nodes)           
        unique_dsts = np.unique(dsts_all)
        self.agg_dst_order = unique_dsts                     
        G = unique_dsts.size

        flows_in  = np.zeros((G,), dtype=int)
        flows_out = np.zeros((G,), dtype=int)
        flows_delay = np.zeros((G,))
        flows_jitter = np.zeros((G,))
        flows_delay_est = np.zeros((G,))
        flows_delay_raw = []
        flows_undeliver = []

        for g, dst in enumerate(unique_dsts):
            f_list = np.where(dsts_all == dst)[0]
            flows_in[g] = int(self.flows_arrivals[f_list, 0:T].sum())

            # 3) 已交付包（按路由方式取不同的“收件箱”）
            if opt > 0:
                # BP: all packets delivered to dst are placed in backlog[dst][dst]
                q = self.backlog[dst][dst].queue
                flows_out[g] = len(q)
                delay_per_pkt = np.array([float(q[i][1] - q[i][0]) for i in range(len(q))], dtype=float)
            else:
                # PH(Ant): packets delivered to dst are placed in backlog_ph[dst][vsink], vsink = self.num_nodes
                q = self.backlog_ph[dst][self.num_nodes].queue
                flows_out[g] = len(q)
                delay_per_pkt = np.array([float(q[i][1] - q[i][0]) for i in range(len(q))], dtype=float)

            flows_delay_raw.append(delay_per_pkt)
            flows_delay[g]  = np.nanmean(delay_per_pkt) if delay_per_pkt.size > 0 else np.nan
            flows_jitter[g] = np.nanvar(delay_per_pkt)  if delay_per_pkt.size > 0 else np.nan

            # 4) Undelivered packets (for estimation)
            undel_list = []
            if opt > 0:
                # BP: packets still in network with destination dst are all in backlog[v][dst]
                for v in range(self.num_nodes):
                    if v == dst:
                        continue
                    qv = self.backlog[v][dst].queue
                    for t0, t1 in qv:
                        undel_list.append(self.T - t0)
            else:
                # PH: packets on all links backlog_ph[v][u], filter cmd == dst
                for v in range(self.num_nodes):
                    _, nb_set = np.nonzero(self.adj_c[v])
                    for u in nb_set:
                        qvu = self.backlog_ph[v][u].queue
                        for pkt in qvu:
                            t0, t1, cmd, last = pkt
                            if cmd == dst:
                                undel_list.append(self.T - t0)

            undel_arr = np.array(undel_list, dtype=float)
            flows_undeliver.append(undel_arr)

            # 5) Estimate delay (delivered + undelivered)
            if undel_arr.size > 0:
                delay_all = np.concatenate([delay_per_pkt, undel_arr])
            else:
                delay_all = delay_per_pkt
            flows_delay_est[g] = np.nanmean(delay_all) if delay_all.size > 0 else np.nan

        return flows_in, flows_out, flows_delay, flows_delay_raw, flows_jitter, flows_undeliver, flows_delay_est

    def plot_pheromones(self, delays, opt, with_labels=True):
        delay_f = np.nan_to_num(delays)
        bbox = self.bbox()
        ces = ['g', 'm']
        for fidx in range(len(self.flows)):
            fig, ax = plt.subplots(1,1)
            for i in range(2):
                j = 1 - i
                f_cnts_p = np.abs(
                    self.pheromones[self.edge_maps[i*self.num_links:(1+i)*self.num_links], fidx]
                )
                f_cnts_n = np.abs(
                    self.pheromones[self.edge_maps[j*self.num_links:(1+j)*self.num_links], fidx]
                )
                f_cnts = np.clip(f_cnts_p - f_cnts_n, 0, None)
                weights = f_cnts  # 10 * f_cnts / (np.amax(f_cnts) + 0.000001) #
                vis_network(
                    self.graph_c,
                    self.src_nodes[fidx:fidx+1],
                    self.dst_nodes[fidx:fidx+1],
                    self.pos_c,
                    weights,
                    delay_f[:, fidx],
                    with_labels,
                    ax=ax,
                    colors=[ces[i], 'r', 'b'],
                    alpha=0.5
                )
            fig_name = "flow_pheromone_visual_{}_f{}_s{}_d{}_cf{:.1f}_opt{}.png".format(
                self.case_name, fidx,
                self.flows[fidx].source_node,
                self.flows[fidx].dest_node,
                self.cf_radius,
                opt)
            fig_name = os.path.join("..", "fig", fig_name)
            ax = plt.gca()
            ax.set_xlim(bbox[0:2])
            ax.set_ylim(bbox[2:4])
            # plt.tight_layout(pad=-0.1)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            plt.close()
            # print("Flow {} plot saved to {}".format(fidx, fig_name))

    def plot_delay(self, delay_n2c, opt):
        for fidx in range(self.num_flows):
            node_colors = ['y' for node in range(self.num_nodes)]
            node_sizes = 10*delay_n2c[:, fidx]
            node_colors[self.src_nodes[fidx]] = 'g'
            node_colors[self.dst_nodes[fidx]] = 'b'
            node_sizes[self.dst_nodes[fidx]] = 400
            ax = nx.draw(
                self.graph_c,
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=True,
                pos=self.pos_c)
            fig_name = "flow_delay_visual_{}_f{}_s{}_d{}_cf{:.1f}_opt{}.png".format(
                self.case_name, fidx,
                self.flows[fidx].source_node,
                self.flows[fidx].dest_node,
                self.cf_radius,
                opt)
            fig_name = os.path.join("..", "fig", fig_name)
            plt.savefig(fig_name, dpi=300)
            plt.close()
            # print("Flow {} plot saved to {}".format(fidx, fig_name))

    def plot_metrics(self, opt):
        arrivals = np.sum(self.flows_arrivals, axis=0)
        pkts_in_network = np.sum(self.flow_pkts_in_network, axis=0)
        departures = np.sum(self.flows_sink_departures, axis=0)

        plt.plot(arrivals)
        plt.plot(departures)
        plt.plot(pkts_in_network)

        plt.suptitle('Departures, Arrivals, and Current amount pkts in network')
        plt.xlabel('T')
        plt.ylabel('the number of packages')
        plt.legend(['Exogenous arrivals', 'Sink departures', 'Pkts in network'], loc='upper right')
        fig_name = "flow_packets_arrivals_per_timeslot_{}_cf{:.1f}_opt_{}.png".format(self.case_name, self.cf_radius, opt)
        fig_name = os.path.join("..", "fig", fig_name)
        plt.savefig(fig_name, dpi=300)
        plt.close()
        # print("Metrics plot saved to {}".format(fig_name))
        return arrivals, pkts_in_network, departures

    def animate_pheromones(self, num_frames, delays, opt, interval, with_labels=True, save_path=True):
        delay_f = np.nan_to_num(delays)
        bbox = self.bbox()
        ces = ['g', 'm']
        links1 = [(u, v) for u, v in self.link_list]
        links2 = [(v, u) for u, v in self.link_list]
        print(f'green direction is {links1}')
        print(f'red direction is {links2}')

        def update(frame, f_show, ax):
            links = links1 + links2
            for f in f_show:
                ax[0, f].clear()
                edge_labels = {}
                for i in range(2):
                    j = 1 - i
                    f_cnts_p = np.abs(
                        self.pheromones_vis[self.edge_maps[i * self.num_links:(1 + i) * self.num_links], f, frame]
                    )
                    f_cnts_n = np.abs(
                        self.pheromones_vis[self.edge_maps[j * self.num_links:(1 + j) * self.num_links], f, frame]
                    )
                    f_cnts = np.clip(f_cnts_p - f_cnts_n, 0, None)
                    weights = f_cnts  # 10 * f_cnts / (np.amax(f_cnts) + 0.000001) #
                    les = self.edge_maps[i * self.num_links:(1 + i) * self.num_links]
                    link_list = [links[ix] for ix in les]
                    for idx, w in enumerate(f_cnts):
                        if i == 0:
                            edge_labels[link_list[idx]] = [round(w, 1)]
                        else:
                            l = link_list[idx][1], link_list[idx][0]
                            edge_labels[l].append(round(w, 1))
                    vis_network(
                        self.graph_c,
                        self.src_nodes[f_show[f]:f_show[f] + 1],
                        self.dst_nodes[f_show[f]:f_show[f] + 1],
                        self.pos_c,
                        weights,
                        delay_f[:, f_show[f]],
                        with_labels,
                        ax=ax[0, f],
                        colors=[ces[i], 'r', 'b'],
                        alpha=0.5
                    )
                # vis_edges(self.graph_c, pos=self.pos_c, edge_labels=edge_labels, ax=ax[0, f], font_size= 7)

                ax[0, f].set_xlim(bbox[0:2])
                ax[0, f].set_ylim(bbox[2:4])
                ax[0, f].set_title(f'flow {f}')

                ax[1, f].clear()
                edge_labels = {}
                # edges = [(idx) for idx in self.graph_c.edges]
                for i in range(2):
                    weights = self.pkt_vis[self.edge_maps[i * self.num_links:(1 + i) * self.num_links], f, frame]
                    les = self.edge_maps[i * self.num_links:(1 + i) * self.num_links]
                    link_list = [links[ix] for ix in les]
                    for idx, w in enumerate(weights):
                        if i == 0:
                            edge_labels[link_list[idx]] = int(w)
                        else:
                            l = link_list[idx][1], link_list[idx][0]
                            if edge_labels[l] == 0:
                                edge_labels[l] = int(w)
                    vis_network(
                        self.graph_c,
                        self.src_nodes[f_show[f]:f_show[f] + 1],
                        self.dst_nodes[f_show[f]:f_show[f] + 1],
                        self.pos_c,
                        weights,
                        delay_f[:, f_show[f]],
                        with_labels,
                        ax=ax[1, f],
                        colors=[ces[i], 'r', 'b'],
                        alpha=0.5
                    )
                    vis_edges(self.graph_c, pos=self.pos_c, edge_labels=edge_labels, ax=ax[1, f])

                ax[1, f].set_xlim(bbox[0:2])
                ax[1, f].set_ylim(bbox[2:4])
                ax[1, f].set_title(f'Number of packets transmitted at time {frame+1}')

        def save_ani():
            ani = animation.FuncAnimation(fig, update, frames=num_frames, fargs=(f_show, ax), interval=interval, repeat=False)
            anim_name = "packets_arrivals_per_timeslot_{}_cf{:.1f}.png".format(self.case_name, self.cf_radius)
            anim_name = os.path.join("..", "fig", anim_name)
            ani.save(anim_name + ".gif", writer="pillow", fps=1, dpi=300)

        def show_ani():
            ani = animation.FuncAnimation(fig, update, frames=num_frames, fargs=(f_show, ax), interval=interval, repeat=False)
            plt.tight_layout()
            plt.show()

        f_show = range(self.num_flows)
        fig, ax = plt.subplots(2, self.num_flows, figsize=(18, 10))

        if save_path:
            save_ani()
        else:
            show_ani()

    def save_pherom(self, opt, id, f_case, items, bursty):

        for idx, t in enumerate(self.t_recordings):
            pherom_name_tim = f'Pheromone_{id}_opt_{opt}_fcase_{f_case}_{t}_{items}_{bursty}.pkl'
            print(f'while saving pherom_name_tim is {pherom_name_tim} and max and min are {np.max(self.pheromones_vis[..., idx+1]), np.min(self.pheromones_vis[..., idx+1])}')
            pherom_name_tim = os.path.join("..", "pkl", pherom_name_tim)
            with open(pherom_name_tim, 'wb') as file:
                pickle.dump(self.pheromones_vis[..., idx+1], file)

    def load_pherom(self, opt, id, f_case, items, t_recording=0, bursty=None, test=False):
        pherom_name_tim = f'Pheromone_{id}_opt_{opt}_fcase_{f_case}_{t_recording}_{items}_{bursty}.pkl'
        pherom_name_tim = os.path.join("..", "pkl", pherom_name_tim)
        if pickle_file_exists(pherom_name_tim):
            if test:
                return pherom_name_tim
            with open(pherom_name_tim, 'rb') as file:
                self.pheromones = pickle.load(file)
            self.pheromones_vis = np.zeros((self.num_di_links, self.num_flows, 1), dtype=float)
            print(f'Load {pherom_name_tim}, max-min: {np.max(self.pheromones), np.min(self.pheromones)}')
            return True
        else:
            return False

