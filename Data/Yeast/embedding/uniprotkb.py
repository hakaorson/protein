import pandas as pd
import re
import networkx as nx
import os
import queue
import random
from goatools import obo_parser


def findSubcellWords(str_input):
    str_remove_head = re.sub('SUBCELLULAR LOCATION: ', "", str_input)
    str_remove_bracket = re.sub('{.*}', "", str_remove_head)
    str_remove_note = re.sub('Note=.*', "", str_remove_bracket)
    str_splited = re.split('\.|;|,', str_remove_note)
    result = []
    for single_str in str_splited:
        single_str = single_str.strip().capitalize()
        if single_str:
            result.append(single_str)
    # print(result)
    return result


def save(datas, path):
    with open(path, 'w') as f:
        for data in datas:
            f.write(data+'\n')


# 读取edge
def read_edges(graph_path):
    nodes, edges = list(), list()
    with open(graph_path) as f:
        for line in f:
            linelist = tuple(line.strip().split('\t'))
            edges.append(linelist)
            for singleid in linelist[:2]:
                nodes.append(singleid)
    return list(set(nodes)), edges


# 读取graph
def read_graph(graph_path):
    nodes, edges = read_edges(graph_path)
    res = nx.Graph()
    for edge in edges:
        if len(edge) == 3:
            res.add_edge(edge[0], edge[1], weight=edge[2])
        else:
            res.add_edge(edge[0], edge[1])
    return res


# 读取mapping
def read_mapping(mapping_path):
    res = {}
    with open(mapping_path) as f:
        for line in f:
            linelist = list(line.strip().split('\t'))
            res[linelist[0]] = linelist[1:]
    return res


def read_uniprotkb(path):
    res = {}
    with open(path, 'r')as f:
        heads = next(f)
        headslist = heads.strip().split('\t')
        enterIndex = headslist.index('Entry')
        seqIndex = headslist.index('Sequence')
        subcellIndex = headslist.index('Subcellular location [CC]')
        goIndex = headslist.index('Gene ontology IDs')
        domainIndex = headslist.index('Cross-reference (Pfam)')
        for line in f:
            linelist = line.split('\t')
            data = {}
            data['seq'] = linelist[seqIndex] if linelist[seqIndex] != '' else []
            data['go'] = linelist[goIndex].replace(' ', '').split(
                ';') if linelist[goIndex] != '' else []
            data['subcell'] = findSubcellWords(
                linelist[subcellIndex]) if linelist[subcellIndex] != '' else []
            data['domain'] = linelist[domainIndex][:-
                                                   1].split(';') if linelist[domainIndex] != '' else []
            res[linelist[enterIndex]] = data
    return res


# 寻找某些节点的某些跳邻居
def findNeisInGraph(graph, nodes, skip):
    nodes_valid = []
    for node in nodes:
        if node in graph.nodes:
            nodes_valid.append(node)
    if len(nodes_valid) == 0:
        return []

    res = nodes_valid
    visited = set(nodes_valid)
    que = queue.Queue()
    for node in nodes_valid:
        que.put(node)
    prenode = res[-1]
    while skip and not que.empty():
        cur_node = que.get()
        cur_neibors = list(graph.neighbors(cur_node))
        for nei in cur_neibors:
            if nei not in visited:
                que.put(nei)
                visited.add(nei)
                res.append(nei)
        if cur_node == prenode:
            prenode = res[-1]
            skip -= 1
    return res


# 计算域相互作用
def compute_edge_feat_domain(graph, v0_domains, v1_domains):
    res = []
    v0_neis_1 = findNeisInGraph(graph, v0_domains, 1)
    v1_neis_1 = findNeisInGraph(graph, v1_domains, 1)
    domain_weight = 0
    direct_linknum = 0
    for v0 in v0_domains:
        for v1 in v1_domains:
            if graph.has_node(v0) and graph.has_node(v1) and graph.has_edge(v0, v1):
                domain_weight += int(graph[v0][v1]['weight'])
                direct_linknum += 1
    res.append(len(set(v0_domains) & set(v1_domains)))  # 两个域有几个相同地
    res.append(direct_linknum)  # 在域图里面链接的数目
    res.append(domain_weight)  # 加权之后的结果
    res.append(len(set(v0_neis_1) & set(v1_neis_1)))  # 一阶共同邻居
    return res


# 计算亚细胞作用
def compute_edge_feat_subcell(mapping, v0_subcellls, v1_subcells):
    res = []
    res.append(len(set(v0_subcellls) & set(v1_subcells)))
    res.append(len(set(v0_subcellls) | set(v1_subcells)))
    # print(res)
    return res


class go_compute():
    def __init__(self, path):
        self.graph = obo_parser.GODag(path)

    def findDirectParent(self, go_name):
        pass

    def computeWangSim(self, v0_go, v1_go):
        temp=self.graph.paths_to_top(v0_go)
        pass

    def compute_edge_feat_go(self, v0_gos, v1_gos):
        '''
        计算go相似性
        '''
        for v0_go in v0_gos:
            for v1_go in v1_gos:
                self.computeWangSim(v0_go, v1_go)


def compute_edge_feats(edges, nodedatas):
    domain_net = read_graph("domain/domain_graph")
    subcell_map = read_mapping("subcell/mapping")
    go_computor = go_compute("go/go-basic.obo")
    res = {}
    for edge in edges:
        v0, v1 = edge
        tempEmb = {}
        tempEmb['domain'] = compute_edge_feat_domain(
            domain_net, nodedatas[v0]['domain'], nodedatas[v1]['domain'])
        tempEmb['subcell'] = compute_edge_feat_subcell(
            subcell_map, nodedatas[v0]['subcell'], nodedatas[v1]['subcell'])
        tempEmb['go'] = go_computor.compute_edge_feat_go(
            nodedatas[v0]['go'], nodedatas[v1]['go'])
        res[edge] = tempEmb
    pass


# 计算blast
def compute_node_feat_blast(mapping, node):
    if node in mapping.keys():
        return list(map(float, mapping[node]))
    else:
        return 420*[0.0]


def compute_node_feats(nodes, nodedatas):
    blast_map = read_mapping("blast/POSSUM_DATA")
    res = {}
    for node in nodes:
        tempEmb = {}
        tempEmb['blast'] = compute_node_feat_blast(blast_map, node)
        res[node] = tempEmb
    pass


if __name__ == "__main__":
    dippath = '../network/dip'
    nodes, edges = read_edges(dippath)
    save(nodes, 'uniprotkb_ids')
    uniprotkb_path = 'uniprotkb_datas'
    uniprotkb_datas = read_uniprotkb(uniprotkb_path)

    compute_edge_feats(edges, uniprotkb_datas)
    compute_node_feats(nodes, uniprotkb_datas)
