import networkx as nx
import matplotlib.pyplot as plt
import random

# Create an empty graph
G = nx.Graph()

# Add starting node
G.add_node('k1')

# Add final node
G.add_node('t+11')

# Generate random nodes and edges
num_nodes = 10  # Number of random nodes
for i in range(num_nodes):
    node = f'node{i+1}'
    G.add_node(node)
    G.add_edge('k1', node)
    G.add_edge(node, 't+11')

# Set positions for the starting and final nodes
pos = {'k1': (0, 0), 't+11': (5, 0)}

# Set positions for random nodes
for i in range(num_nodes):
    node = f'node{i+1}'
    x = random.uniform(0, 5)
    y = random.uniform(-1, 1)
    pos[node] = (x, y)

# Draw the network
nx.draw_networkx_nodes(G, pos=pos, nodelist=['k1', 't+11'], node_color='orange',  node_size=500,edgecolors='black')
nx.draw_networkx_labels(G, pos=pos, labels={'k1': '', 't+11': ''})

nx.draw_networkx_edges(G, pos=pos)
nx.draw_networkx_labels(G, pos=pos, labels={node: '' for node in G.nodes() if node not in ['k1', 't+11']})

nx.draw_networkx_nodes(G, pos=pos, nodelist=[node for node in G.nodes() if node not in ['k1', 't+11']], edgecolors='black')

# Set plot title
plt.title('Network Graph')

# Show the plot
plt.show()
