import sys
# import EoN
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import pickle

import numbers
from collections import Counter
from networkx.utils import discrete_sequence, py_random_state, weighted_choice
import numpy as np

@py_random_state(7)
def scale_free_graph(
    n,
    alpha=0.41,
    beta=0.54,
    gamma=0.05,
    delta_in=0.2,
    delta_out=0,
    create_using=None,
    seed=None,
):
    """Returns a scale-free directed graph.

    Parameters
    ----------
    n : integer
        Number of nodes in graph
    alpha : float
        Probability for adding a new node connected to an existing node
        chosen randomly according to the in-degree distribution.
    beta : float
        Probability for adding an edge between two existing nodes.
        One existing node is chosen randomly according the in-degree
        distribution and the other chosen randomly according to the out-degree
        distribution.
    gamma : float
        Probability for adding a new node connected to an existing node
        chosen randomly according to the out-degree distribution.
    delta_in : float
        Bias for choosing nodes from in-degree distribution.
    delta_out : float
        Bias for choosing nodes from out-degree distribution.
    create_using : NetworkX graph constructor, optional
        The default is a MultiDiGraph 3-cycle.
        If a graph instance, use it without clearing first.
        If a graph constructor, call it to construct an empty graph.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Examples
    --------
    Create a scale-free graph on one hundred nodes::

    >>> G = nx.scale_free_graph(100)

    Notes
    -----
    The sum of `alpha`, `beta`, and `gamma` must be 1.

    References
    ----------
    .. [1] B. Bollobás, C. Borgs, J. Chayes, and O. Riordan,
           Directed scale-free graphs,
           Proceedings of the fourteenth annual ACM-SIAM Symposium on
           Discrete Algorithms, 132--139, 2003.
    """

    def _choose_node(candidates, node_list, delta):
        if delta > 0:
            bias_sum = len(node_list) * delta
            p_delta = bias_sum / (bias_sum + len(candidates))
            if seed.random() < p_delta:
                return seed.choice(node_list)
        return seed.choice(candidates)

    if create_using is None or not hasattr(create_using, "_adj"):
        # start with 3-cycle
        G = nx.empty_graph(3, create_using, default=nx.DiGraph)
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    else:
        G = create_using
    # if not (G.is_directed() and G.is_multigraph()):
        # raise nx.NetworkXError("MultiDiGraph required in create_using")

    if alpha <= 0:
        raise ValueError("alpha must be > 0.")
    if beta <= 0:
        raise ValueError("beta must be > 0.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    if abs(alpha + beta + gamma - 1.0) >= 1e-9:
        raise ValueError("alpha+beta+gamma must equal 1.")

    if delta_in < 0:
        raise ValueError("delta_in must be >= 0.")

    if delta_out < 0:
        raise ValueError("delta_out must be >= 0.")

    # pre-populate degree states
    vs = sum([count * [idx] for idx, count in G.out_degree()], [])
    ws = sum([count * [idx] for idx, count in G.in_degree()], [])

    # pre-populate node state
    node_list = list(G.nodes())

    # see if there already are number-based nodes
    numeric_nodes = [n for n in node_list if isinstance(n, numbers.Number)]
    if len(numeric_nodes) > 0:
        # set cursor for new nodes appropriately
        cursor = max([int(n.real) for n in numeric_nodes]) + 1
    else:
        # or start at zero
        cursor = 0

    while len(G) < n:
        r = seed.random()

        # random choice in alpha,beta,gamma ranges
        if r < alpha:
            # alpha
            # add new node v
            v = cursor
            cursor += 1
            # also add to node state
            node_list.append(v)
            # choose w according to in-degree and delta_in
            w = _choose_node(ws, node_list, delta_in)

        elif r < alpha + beta:
            # beta
            # choose v according to out-degree and delta_out
            v = _choose_node(vs, node_list, delta_out)
            # choose w according to in-degree and delta_in
            w = _choose_node(ws, node_list, delta_in)
            
            if len(G.edges) < ((len(G.nodes)-2)**2):
                while (G.has_edge(v, w)):
                    # choose v according to out-degree and delta_out
                    v = _choose_node(vs, node_list, delta_out)
                    # choose w according to in-degree and delta_in
                    w = _choose_node(ws, node_list, delta_in)
                    print(len(G.edges), len(G.nodes), v,w)

        else:
            # gamma
            # choose v according to out-degree and delta_out
            v = _choose_node(vs, node_list, delta_out)
            # add new node w
            w = cursor
            cursor += 1
            # also add to node state
            node_list.append(w)

        # add edge to graph
        G.add_edge(v, w)

        # update degree states
        vs.append(v)
        ws.append(w)

    return G

# Here goes the state of graph
# N = 500
N = int(sys.argv[2])
beta = float(sys.argv[3])
alpha = (1.0-beta)/4
gamma = alpha*3
# G = nx.fast_gnp_random_graph(N, 0.02, directed=True)
# G = nx.scale_free_graph(N, alpha=alpha, beta=beta, gamma=gamma, delta_in=0., delta_out=0.)
# modified version, erase multigraph requirement
G = scale_free_graph(N, alpha=alpha, beta=beta, gamma=gamma, delta_in=0.1, delta_out=0.1)

#they will vary in the rate of leaving exposed class.
#and edges will vary in transition rate.
#there is no variation in recovery rate.

node_attribute_dict = {node: 0.5+random.random() for node in G.nodes()}
edge_attribute_dict = {edge: 0.5+random.random() for edge in G.edges()}

# print(edge_attribute_dict)

# logistic func will be set according to number of followers(here is outdegree)
logistic_func_scale = np.zeros(len(G.nodes))
# user_cost_scale = np.zeros(len(G.nodes))
outdegrees = G.out_degree
print(outdegrees)
for i, deg in outdegrees:
    logistic_func_scale[i] = deg
    

user_cost_scale = logistic_func_scale / max(logistic_func_scale) * 9
logistic_func_scale = logistic_func_scale/max(logistic_func_scale) * 2

print(len(G.edges), len(G.nodes))

logistic_func_param_dict = {node: 1.+logistic_func_scale[node] for node in G.nodes()}
user_cost = {node: 1.+user_cost_scale[node] for node in G.nodes()}
print(min(list(user_cost.values())), max(list(user_cost.values())), np.mean(list(user_cost.values())))
# in synthetic env, intensity will randomly given
init_intensity_dict = {node: 0.5+random.random() for node in G.nodes()}

nx.set_node_attributes(G, values=node_attribute_dict, name='expose2infect_weight')

nx.set_node_attributes(G, values=logistic_func_param_dict, name='logistic_weight')

nx.set_node_attributes(G, values=init_intensity_dict, name='intensity_weight')

nx.set_edge_attributes(G, values=edge_attribute_dict, name='transmission_weight')


H = nx.DiGraph()
H.add_node('S')
H.add_edge('EI', 'I', rate = 0.8, weight_label='expose2infect_weight')
H.add_edge('ER', 'R', rate = 0.8, weight_label='expose2infect_weight')
# H.add_edge('I', 'R', rate = 0.1)

J = nx.DiGraph()
J.add_edge(('I', 'S'), ('I', 'EI'), rate = 0.5, weight_label='transmission_weight')
J.add_edge(('R', 'S'), ('R', 'ER'), rate = 0.5, weight_label='transmission_weight')
J.add_edge(('I', 'R'), ('I', 'I'), rate = 0.2, weight_label='transmission_weight')
J.add_edge(('R', 'I'), ('R', 'R'), rate = 0.2, weight_label='transmission_weight')

# IC = defaultdict(lambda: 'S')
IC = {}
for i in range(N):
    IC[i] = 'S'
for node in range(20):
    IC[node] = 'I'

return_statuses = ('S', 'EI', 'ER', 'I', 'R')

fo = open(sys.argv[1], "wb")
pickle.dump([N, G, H, J, IC, return_statuses, user_cost], fo)
#nx.write_gpickle(G, sys.argv[1]+".G")
#nx.write_gpickle(H, sys.argv[1]+".H")
#nx.write_gpickle(J, sys.argv[1]+".J")
