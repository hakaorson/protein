import pandas as pd
import math
import sys
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
        self.graph = obo_parser.GODag(path, optional_attrs="relationship")
        self.computed_SV = {}
        self.lin_static = {}

    def findDirectParent(self, go_term):
        is_a_parents = list(go_term.parents)
        part_of_parents = list(
            go_term.relationship['part_of']) if 'part_of' in go_term.relationship.keys() else []
        return is_a_parents, part_of_parents

    def extract_graph(self, go_term):
        '''
        从go注释网络里面提取局部网络
        '''
        nodes = set([go_term.id])
        edges = dict()
        visited = set()
        que = queue.Queue()
        que.put(go_term)
        while not que.empty():
            cur_node = que.get()
            if cur_node in visited:
                continue
            visited.add(cur_node)
            is_a_parents, part_of_parents = self.findDirectParent(cur_node)
            for part_of_p in part_of_parents:
                que.put(part_of_p)
                edges[(cur_node.id, part_of_p.id)] = 0.6
                nodes.add(part_of_p.id)
            for is_a_p in is_a_parents:
                que.put(is_a_p)
                edges[(cur_node.id, is_a_p.id)] = 0.8
                nodes.add(is_a_p.id)
        nodemap = {}
        for index, node in enumerate(nodes):
            nodemap[node] = index
        matrix = [[0 for j in range(len(nodemap))]
                  for i in range(len(nodemap))]
        for edge in edges.keys():
            v0, v1 = nodemap[edge[0]], nodemap[edge[1]]
            matrix[v0][v1] = edges[edge]
        res_nodes = list(nodes)
        res_edges = [[edge[0], edge[1], edges[edge]] for edge in edges]
        return res_nodes, res_edges

    def compute_sv(self, go):
        if go in self.computed_SV:
            return self.computed_SV[go]
        if go not in self.graph.keys():
            return {}
        begin = self.graph.query_term(go)
        nodes, edges = self.extract_graph(begin)
        # 为拓扑排序汇集所有的边信息和节点信息
        edge_in_num = {}
        edges_sum = {}
        node_res = {}
        for node in nodes:
            node_res[node] = 0.0
            edge_in_num[node] = 0
        node_res[begin.id] = 1.0
        for v0, v1, w in edges:
            edge_in_num[v1] += 1
            if v0 in edges_sum.keys():
                edges_sum[v0].append([v1, w])
            else:
                edges_sum[v0] = [[v1, w]]
        que = queue.Queue()
        que.put(begin.id)
        # 执行拓扑排序算法
        while not que.empty():
            cur = que.get()
            if cur in edges_sum.keys():
                for parent, w in edges_sum[cur]:
                    edge_in_num[parent] -= 1
                    node_res[parent] = max(node_res[parent], node_res[cur]*w)
                    if edge_in_num[parent] == 0:
                        que.put(parent)
        self.computed_SV[go] = node_res
        return node_res

    def computeWangSim(self, v0_go, v1_go):
        v0_SV = self.compute_sv(v0_go)
        v1_SV = self.compute_sv(v1_go)
        sum_v0, sum_v1, sum_com = 0, 0, 0
        commons = v0_SV.keys() & v1_SV.keys()
        if len(commons) == 0:
            return 0
        for com in commons:
            sum_com += v0_SV[com]
            sum_com += v1_SV[com]
        for v0 in v0_SV.keys():
            sum_v0 += v0_SV[v0]
        for v1 in v1_SV.keys():
            sum_v1 += v1_SV[v1]
        return sum_com/(sum_v0+sum_v1)

    def static_allgo_info(self, go_info):
        for go in go_info:
            if go in self.lin_static.keys():
                self.lin_static[go] += 1
            else:
                self.lin_static[go] = 1

    def computeLinSim(self, v0_gos, v1_gos):
        v0_parents, v1_parents = set(), set()
        for v0_go in v0_gos:
            v0_query = self.graph.query_term(v0_go)
            if v0_query:
                v0_parents = v0_parents | set(v0_query.parents)
        for v1_go in v1_gos:
            v1_query = self.graph.query_term(v1_go)
            if v1_query:
                v1_parents = v1_parents | set(v1_query.parents)
        common_parents = v0_parents & v1_parents & self.lin_static.keys()  # TODO 只统计了当前的蛋白质特征
        min_common = sys.maxsize
        max_v0, max_v1 = 0, 0
        for cpa in common_parents:
            min_common = min(min_common, self.lin_static[cpa])
        for v0_go in v0_gos:
            max_v0 = max(max_v0, self.lin_static[v0_go])
        for v1_go in v1_gos:
            max_v1 = max(max_v1, self.lin_static[v1_go])
        if max_v0 == 0 or max_v1 == 0 or min_common == sys.maxsize:
            return 0
        return 2*math.log(1/min_common)/(math.log(1/max_v0)+math.log(1/max_v1))

    def compute_edge_feat_go(self, v0_gos, v1_gos):
        '''
        计算go相似性
        '''
        res = []
        '''
        wang相似性，取自徐斌师兄的论文
        两个蛋白质拆尊尊自己的go注释
        计算其中任意两个之间的go相似性
        
        使用go图可以提取go的关系，其中parent是直接关系（表示is_a），而在relationship中的part_of关键字表示的是part_of的关系
        具体的计算过程可以从论文中得出
        '''
        matrix = [[0 for j in range(len(v1_gos)+1)]
                  for i in range(len(v0_gos)+1)]
        for v0_index, v0_go in enumerate(v0_gos):
            for v1_index, v1_go in enumerate(v1_gos):
                matrix[v0_index][v1_index] = self.computeWangSim(v0_go, v1_go)
                matrix[v0_index][-1] = max(matrix[v0_index]
                                           [-1], matrix[v0_index][v1_index])
                matrix[-1][v1_index] = max(matrix[-1]
                                           [v1_index], matrix[v0_index][v1_index])
        temp_sum = 0
        for i in range(len(v0_gos)):
            temp_sum += matrix[i][-1]
        for j in range(len(v1_gos)):
            temp_sum += matrix[-1][j]
        temp_sum = temp_sum/(len(v0_gos)+len(v1_gos)
                             ) if len(v0_gos) or len(v1_gos) else 0
        res.append(temp_sum)
        '''
        lin相似性，取自徐斌论文第五章
        注意计算中所有的蛋白质只是网络里面涉及到的蛋白质
        '''
        res.append(self.computeLinSim(v0_gos, v1_gos))
        '''
        其他特征
        '''
        common_go_nums = len(set(v0_gos) & set(v1_gos))
        all_go_nums = len(set(v0_gos) | set(v1_gos))
        res.append(common_go_nums)
        res.append(all_go_nums)
        return res


def compute_edge_feats(edges, nodedatas):
    domain_net = read_graph("domain/domain_graph")
    subcell_map = read_mapping("subcell/mapping")
    go_computor = go_compute("go/go-basic.obo")
    for node in nodedatas.keys():
        go_computor.static_allgo_info(nodedatas[node]['go'])
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
    return res


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
    return res


if __name__ == "__main__":
    dippath = '../network/dip'
    nodes, edges = read_edges(dippath)
    save(nodes, 'uniprotkb_ids')
    uniprotkb_path = 'uniprotkb_datas'
    uniprotkb_datas = read_uniprotkb(uniprotkb_path)

    edge_feats = compute_edge_feats(edges, uniprotkb_datas)
    node_feats = compute_node_feats(nodes, uniprotkb_datas)

    dip_node_path = 'dip_node'
    dip_edge_path = 'dip_edge'
    with open(dip_edge_path, 'w') as f:
        for edge in edge_feats.keys():
            datas = []
            dict_datas = edge_feats[edge]
            for key in dict_datas.keys():
                datas.extend(dict_datas[key])
            datas = list(map(float, datas))
            short_datas = list(map(lambda num: "{:.2f}".format(num), datas))
            strings = edge[0]+'\t' + edge[1]+'\t'+'\t'.join(short_datas)+'\n'
            f.write(strings)
    with open(dip_node_path, 'w') as f:
        for node in node_feats.keys():
            datas = []
            dict_datas = node_feats[node]
            for key in dict_datas.keys():
                datas.extend(dict_datas[key])
            datas = list(map(float, datas))
            short_datas = list(map(lambda num: "{:.2f}".format(num), datas))
            strings = node+'\t'+'\t'.join(short_datas)+'\n'
            f.write(strings)
