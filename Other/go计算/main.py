from goatools import obo_parser
import queue
import math
import sys


class go_compute():
    def __init__(self, obopath, allproteingopath):
        self.graph = obo_parser.GODag(obopath, optional_attrs="relationship")
        self.computed_SV = {}
        self.lin_static, self.allproteinnum = self.static_allgo_info(
            allproteingopath)

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
        if go in self.computed_SV.keys():
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

    def computeWangSimSingle(self, v0_go, v1_go):
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

    def computeWangSim(self, v0_gos, v1_gos):
        matrix = [[0 for j in range(len(v1_gos)+1)]
                  for i in range(len(v0_gos)+1)]
        for v0_index, v0_go in enumerate(v0_gos):
            for v1_index, v1_go in enumerate(v1_gos):
                matrix[v0_index][v1_index] = self.computeWangSimSingle(
                    v0_go, v1_go)
                matrix[v0_index][-1] = max(matrix[v0_index]
                                           [-1], matrix[v0_index][v1_index])
                matrix[-1][v1_index] = max(matrix[-1]
                                           [v1_index], matrix[v0_index][v1_index])
        temp_sum = 0
        for i in range(len(v0_gos)):
            temp_sum += matrix[i][-1]
        for j in range(len(v1_gos)):
            temp_sum += matrix[-1][j]
        res = temp_sum/(len(v0_gos)+len(v1_gos)
                        ) if len(v0_gos) or len(v1_gos) else 0
        return res

    def static_allgo_info(self, allgopath):
        static = {}
        proteinnum = 0
        with open(allgopath, 'r')as f:
            next(f)
            for line in f:
                proteinnum += 1
                linedatas = line.strip().split('\t')
                gos = list(linedatas[1].split(';')) if len(
                    linedatas) > 1 else []
                for go in gos:
                    static[go] = static.get(go, 0)+1
        return static, proteinnum

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
        common_parents = v0_parents & v1_parents
        common_parents = [item.id for item in common_parents]

        allkeys = self.lin_static.keys()
        min_common = sys.maxsize
        max_v0, max_v1 = 0, 0
        for cpa in common_parents:
            min_common = min(
                min_common, self.lin_static[cpa] if cpa in allkeys else sys.maxsize)
        for v0_go in v0_gos:
            max_v0 = max(
                max_v0, self.lin_static[v0_go] if v0_go in allkeys else 0)
        for v1_go in v1_gos:
            max_v1 = max(
                max_v1, self.lin_static[v1_go] if v1_go in allkeys else 0)
        if max_v0 == 0 or max_v1 == 0 or min_common == sys.maxsize:
            return 0
        return 2*math.log(min_common/self.allproteinnum)/(math.log(max_v0/self.allproteinnum)+math.log(max_v1/self.allproteinnum))

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
        res.append(self.computeWangSim(v0_gos, v1_gos))
        '''
        lin相似性，取自徐斌论文第五章
        注意计算中所有的蛋白质只是网络里面涉及到的蛋白质
        '''
        res.append(self.computeLinSim(v0_gos, v1_gos))
        '''
        论文Predicting protein complex in protein interaction network - a supervised learning based method提供了一种go特征的计算方法
        '''
        '''
        其他特征
        '''
        common_go_nums = len(set(v0_gos) & set(v1_gos))
        all_go_nums = len(set(v0_gos) | set(v1_gos))
        res.append(common_go_nums)
        res.append(all_go_nums)
        return res


if __name__ == "__main__":
    gos0 = ['GO:0003723', 'GO:0005637', 'GO:0005739', 'GO:0005783', 'GO:0006614',
            'GO:0006620', 'GO:0008320', 'GO:0016021', 'GO:0031204', 'GO:0031207', 'GO:0046967']
    gos1 = ['GO:0000324', 'GO:0005635',
            'GO:0005768', 'GO:0016021', 'GO:0043328']
    go_computor = go_compute(
        'go-basic.obo', 'uniprot-filtered-reviewed_yes.tab')
    wang = go_computor.computeWangSim(gos0, gos1)
    lin = go_computor.computeLinSim(gos0, gos1)
    pass
