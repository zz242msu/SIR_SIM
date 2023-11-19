import networkx as nx
import ndlib
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from time import time
import matplotlib.pyplot as plt
import torch_geometric.datasets as ds
import random
from torch_geometric.datasets import Planetoid

def connSW(beta=0.1):
    n = random.randint(1000, 1500)  # Randomize size between 1000 and 1500
    k = 10  # Number of nearest neighbors in the ring topology

    # Ensure n is greater than k
    if n <= k:
        n = k + 1

    g = nx.connected_watts_strogatz_graph(n, k, 0.1)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        if beta:
            weight = beta
        g[a][b]['weight'] = weight
        config.add_edge_configuration("threshold", (a, b), weight)

    return g, config

def BA():
    g = nx.barabasi_albert_graph(1000, 5)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        g[a][b]['weight'] = weight
        config.add_edge_configuration("threshold", (a, b), weight)

    return g, config

def ER():

    g = nx.erdos_renyi_graph(5000, 0.002)

    while nx.is_connected(g) == False:
        g = nx.erdos_renyi_graph(5000, 0.002)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    return g, config

def CiteSeer():
    dataset = Planetoid(root='./Planetoid', name='CiteSeer')  # Cora, CiteSeer, PubMed
    data = dataset[0]
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)

    c = max(nx.connected_components(G), key=len)
    g = G.subgraph(c).copy()
    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    return g, config

def PubMed():
    dataset = Planetoid(root='./Planetoid', name='PubMed')  # Cora, CiteSeer, PubMed
    data = dataset[0]
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)

    c = max(nx.connected_components(G), key=len)
    g = G.subgraph(c).copy()
    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    return g, config

def Cora():
    dataset = Planetoid(root='./Planetoid', name='Cora')  # Cora, CiteSeer, PubMed
    data = dataset[0]
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)

    c = max(nx.connected_components(G), key=len)
    g = G.subgraph(c).copy()
    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(40,80)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    return g, config

def photo():

    dataset = ds.Amazon(root='./geo', name = 'Photo')
    data = dataset[0]
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)
    g = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(5,20)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    return g, config

def coms():

    dataset = ds.Amazon(root='./geo', name = 'Computers')
    data = dataset[0]
    edges = (data.edge_index.numpy()).T.tolist()
    G = nx.from_edgelist(edges)
    g = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    config = mc.Configuration()

    for a, b in g.edges():
        weight = random.randrange(5,20)
        weight = round(weight / 100, 2)
        config.add_edge_configuration("threshold", (a, b), weight)
        g[a][b]['weight'] = weight

    return g, config

# Function to run SIR model and save graph
def run_and_save_sir_model(graph_func, graph_name, run_number, graph_args=[], beta=0.1, gamma=0.01, seed=42, steps=10):
    G, config = graph_func(*graph_args)
    graph_size = len(G)  # Get the size of the graph

    # Model selection
    model = ep.SIRModel(G)

    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter('beta', beta)
    config.add_model_parameter('gamma', gamma)

    # Set the initial infected node
    random.seed(seed)
    initial_infected = random.choice([n for n, d in G.degree() if d == max(dict(G.degree()).values())])
    config.add_model_initial_configuration("Infected", [initial_infected])
    
    model.set_initial_status(config)

    # Simulation execution
    iterations = model.iteration_bunch(steps)

    # Write iterations to a file. Only record infected_nodes
    # Update file saving to include graph size in the filename
    with open(f'infected_nodes_{graph_name}_{graph_size}_run{run_number}.txt', 'w') as file:
        file.write(f"{graph_name} size {graph_size} run {run_number} - Infected nodes:\n")
        infected_nodes = [n for n in G.nodes if G.nodes[n]['status'] == 1]
        file.write(str(infected_nodes) + "\n")

    # Update the graph with the status from the last iteration
    for i, node_status in model.status.items():
        G.nodes[i]['status'] = node_status

    # Node colors based on status
    status_colors = {0: 'green', 1: 'red', 2: 'blue'}
    colors = [status_colors[node[1]['status']] for node in G.nodes(data=True)]

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=colors, with_labels=False, node_size=20)

    # Save the plot with run number in the filename
    plt.savefig(f'graph_infected_state_{graph_name}_run{run_number}.png', format='PNG')
    plt.close()

# List of graph functions, their names, and specific arguments
graphs = [
    (connSW, "connSW", [0.1]),  # random size
    (BA, "BA", []),
    (ER, "ER", []),
    (CiteSeer, "CiteSeer", []),
    (PubMed, "PubMed", []),
    (Cora, "Cora", []),
    (photo, "photo", []),
    (coms, "coms", [])
]

# Run and save for each graph type 20 times
for graph_func, graph_name, graph_args in graphs:
    for run_number in range(1, 21):  # Run 20 times
        run_and_save_sir_model(graph_func, graph_name, run_number, graph_args)
