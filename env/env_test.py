import EoN
import networkx as nx
import matplotlib.pyplot as plt
import random

N=10000
G = nx.fast_gnp_random_graph(N, 5/(N-1.))



#set up the code to handle constant transmission rate
#with fixed recovery time.
def trans_time_fxn(source, target, rate):
    return random.expovariate(rate)

def rec_time_fxn(node,D):
    return D

D = 5
tau = 0.3
initial_inf_count = 100

t, S, I, R = EoN.fast_nonMarkov_SIR(G,
                        trans_time_fxn=trans_time_fxn,
                        rec_time_fxn=rec_time_fxn,
                        trans_time_args=(tau,),
                        rec_time_args=(D,),
                        initial_infecteds = range(initial_inf_count))

print(t.shape)
print(t[:10])
print(S.shape, I.shape, R.shape)
print(S[:10], I[:10], R[:10])
# note the comma after ``tau`` and ``D``.  This is needed for python
# to recognize these are tuples

# initial condition has first 100 nodes in G infected.
