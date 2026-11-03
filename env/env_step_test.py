import EoN
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import random

N = 1000
G = nx.fast_gnp_random_graph(N, 5./(N-1), directed=True)

#they will vary in the rate of leaving exposed class.
#and edges will vary in transition rate.
#there is no variation in recovery rate.

#node_attribute_dict = {node: 0.5+random.random() for node in G.nodes()}
#edge_attribute_dict = {edge: 0.5+random.random() for edge in G.edges()}

node_attribute_dict = {node: 0.2 for node in G.nodes()}
edge_attribute_dict = {edge: 0.2 for edge in G.edges()}

nx.set_node_attributes(G, values=node_attribute_dict, name='expose2infect_weight')
nx.set_edge_attributes(G, values=edge_attribute_dict, name='transmission_weight')


H = nx.DiGraph()
H.add_node('S')
H.add_edge('E', 'I', rate = 0.6, weight_label='expose2infect_weight')
H.add_edge('I', 'R', rate = 0.1)

J = nx.DiGraph()
J.add_edge(('I', 'S'), ('I', 'E'), rate = 0.2, weight_label='transmission_weight')
J.add_edge(('I', 'R'), ('I', 'I'), rate = 0.1, weight_label='transmission_weight')
J.add_edge(('R', 'I'), ('R', 'R'), rate = 0.1, weight_label='transmission_weight')

for n, nbrs in J.adj.items():
    print(n)
    print(nbrs)

IC = defaultdict(lambda: 'S')
for node in range(200):
    IC[node] = 'I'

return_statuses = ('S', 'E', 'I', 'R')

t, IC, S, E, I, R = EoN.Gillespie_simple_contagion(G, H, J, IC, return_statuses,
                                        tmin=0, tmax = 10)

plt.semilogy(t, S, label = 'Susceptible')
plt.semilogy(t, E, label = 'Exposed')
plt.semilogy(t, I, label = 'Infected')
plt.semilogy(t, R, label = 'Recovered')
plt.legend()

#plt.show()
plt.savefig('SEIR_1k_n02e02.png')

# modify infect rate
H['E']['I']['rate'] = 0.0
J[('I', 'S')][('I', 'E')]['rate'] = 0.0

for n, nbrs in J.adj.items():
    print(n)
    print(nbrs)

t, IC, S, E, I, R = EoN.Gillespie_simple_contagion(G, H, J, IC, return_statuses,
                                        tmin=10, tmax = 20)




plt.semilogy(t, S, label = 'Susceptible')
plt.semilogy(t, E, label = 'Exposed')
plt.semilogy(t, I, label = 'Infected')
plt.semilogy(t, R, label = 'Recovered')
plt.legend()

#plt.show()
plt.savefig('SEIR_1k_n02e02_noe.png')

graph_hist, IC = EoN.Gillespie_simple_contagion(G, H, J, IC, return_statuses, tmin=20, tmax = 30, return_full_data=True)

print(graph_hist.get_statuses(time=22.))
print(graph_hist.get_statuses(time=30.))

