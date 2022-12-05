import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from numpy import array
import copy

# Initialize the original graph and the seed for initial layout
# Example 1
i = [0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 14, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20]
j = [1, 2, 0, 3, 0, 1, 4, 3, 5, 7, 8, 9, 4, 6, 5, 7, 8, 4, 6, 4, 6, 4, 10, 9, 11, 10, 12, 11, 16, 14, 15, 13, 16, 17, 13, 12, 14, 14, 18, 17, 19, 18, 20, 19]

# Example 2
#i = [0, 1, 2, 3, 4, 5, 6, 0, 3, 8, 4, 10, 1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 11]
#j = [1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 0, 3, 8, 4, 10]

# Example 3
#i = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 7]
#j = [1, 2, 3, 4, 5, 6, 0, 7, 0, 1, 2, 3, 4, 5, 6, 0]

seed=28
max_iter=10

# Constructing the graph layout based on the initialization
m= max(max(i),max(j))+1
v =  [1 for j in range(len(i))]
A = sparse.coo_matrix((v,(i,j)),shape=(m,m)).toarray()
G=nx.from_numpy_matrix(A)
layout = nx.spring_layout(G, seed=seed)

# Iterative coarsening of the graph via spectral grouping based on eigenvector centrality measure. This follows:
# Node clustering for material science: Webb, Michael A., Jean-Yves Delannoy, and Juan J. De Pablo.
# "Graph-based approach to systematic molecular coarse-graining."
# Journal of chemical theory and computation 15, no. 2 (2018): 1199-1208.

Ak = A

count = 0
while count < max_iter:  # Set a iteration limit to avoid indefinite loop
    # First step is to visualize the graph at the current iteration based on eigenvector centrality measure
    Gk = nx.from_numpy_matrix(Ak)
    nodes = Gk.nodes()
    edges = copy.deepcopy(Gk.edges())
    score = list(nx.eigenvector_centrality(Gk).values())
    plt.figure(count, figsize=(20, 20))
    plt.title("Graph coarsening by spectral grouping, Iteration: " + str(count))
    nx.draw_networkx_labels(Gk, pos=layout, font_size=12, font_color='white', font_family='sans-serif')
    nx.draw_networkx_nodes(Gk, nodelist=nodes, label=nodes, node_color=score, pos=layout, node_size=500)
    nx.draw_networkx_edges(Gk, edgelist=edges, pos=layout, edge_color='magenta', width=4)
    plt.axis([-1.05, 1.05, -1.05, 1.05])
    plt.show()

    # Termination criteria is when the graph becomes a lone node
    if len(Ak) == 1:
        break

    # This is the main body of the code where coarsening grouping is done via spectral grouping following Webb et al.
    sscore = np.argsort(score)
    ind = np.zeros(len(nodes))
    roll = 0
    for a in sscore:
        if ind[a] == 0:
            roll = roll + 1
            broll = roll
            ind[a] = broll
        else:
            broll = ind[a]
        nlist = [b for b in Gk.neighbors(a)]
        nscore = min([score[b] for b in nlist])
        for c in nlist:
            if (ind[c] == 0) and (score[c] >= nscore):
                ind[c] = broll
    ind = [int(k - 1) for k in ind]

    print(ind)

    # This is the section where the trimmed adjacency matrix is created following the coarsened grouping
    for k in range(roll):
        I = [ind[k] for k in i]
        J = [ind[k] for k in j]
    V = [1 for J in range(len(I))]
    Ak = sparse.coo_matrix((V, (I, J)), shape=(roll, roll)).toarray()
    Ak = Ak - np.diag(np.diag(Ak))
    i = I
    j = J

    # This is the section where the earlier layout is modified following a centroid rule. A weighted centroid rule could also be used instead.
    centroids = []
    for k in range(roll):
        t = [j == k for j in ind]
        centroids.append(sum([score[i] * layout[i] for i, x in enumerate(t) if x]) / sum(
            [score[i] for i, x in enumerate(t) if x]))  # Replace score[i] by 1 for simple centroid rule

    layout = dict(zip([j for j in range(roll)], centroids))
    count = count + 1
