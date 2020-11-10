import pandas as pd
import random
import networkx as nx


def read_graph(node_path, edge_path):
    res = nx.Graph()
    with open(node_path, 'r') as f:
        for nodedata in f:
            nodedata_splited = nodedata.split('\t')
            node_id = nodedata_splited[0]
            node_feat = list(map(float, nodedata_splited[1:]))
            res.add_node(node_id, w=node_feat)
    with open(edge_path, 'r')as f:
        for edgedata in f:
            edgedata_splited = edgedata.split('\t')
            v0, v1 = edgedata_splited[:2]
            edge_feat = list(map(float, edgedata_splited[2:]))
            res.add_edge(v0, v1, w=edge_feat)
    return res


def get_single_random_graph_nodes(graph, size):
    beginer = random.choice(list(graph.nodes.keys()))
    node_set = set([beginer])
    neighbor_sets = set(graph.neighbors(beginer))
    while len(node_set) < size:
        try:
            next_node = random.choice(list(neighbor_sets))
            neighbor_sets.remove(next_node)
            node_set.add(next_node)
            for nei in graph.neighbors(next_node):
                if nei not in node_set:
                    neighbor_sets.add(nei)
        except Exception:
            return None
    return tuple(node_set)


def read_bench(path):
    res = set()
    with open(path, 'r')as f:
        for line in f:
            line_splited = line.split('\t')
            res.add(tuple(line_splited))
    return res


if __name__ == "__main__":
    node_path = "Yeast/embedding/dip_node"
    edge_path = "Yeast/embedding/dip_edge"
    postive_path = "Yeast/bench/CYC2008"
    middle_path = "Yeast/bench/dip_coach"
    graph = read_graph(node_path, edge_path)
    get_single_random_graph_nodes(graph, 5)
    bench_data = read_bench(postive_path)
