from Data.Yeast import data
from Model import graph_classify
from Model import model
import random
import torch
import time
import pickle
import os
import subprocess
from Check import metrix

PY2_path = "D:/software/Anaconda/envs/python2.7/python.exe"


class baseMethod():
    def __init__(self, method_path, graph_path, res_path):
        self.method_path = method_path
        self.graph_path = graph_path
        self.res_path = res_path

    def main(self, reload=True):
        if reload or not os.path.exists(self.res_path):
            cmd_res = self.run()
            self.getcomplexes(cmd_res)
        return data.read_bench(self.res_path)

    def run(self):
        assert(os.path.exists(self.method_path))
        cmd = "{} {} {}".format(PY2_path, self.method_path, self.graph_path)
        # print(cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        res = []
        while p.poll() is None:
            line = p.stdout.readline().decode("utf8")
            res.append(line)
        return res

    def getcomplexes(self, cmd_res):
        ImportError


class ipca_method(baseMethod):
    def __init__(self, method_path, graph_path, res_path):
        super().__init__(method_path, graph_path, res_path)

    def getcomplexes(self, cmd_res):
        with open(self.res_path, 'w') as f:
            for index, item in enumerate(cmd_res):
                if index % 2 == 0:
                    f.write(item)
                    # print(item)


class dpclus_method(baseMethod):
    def __init__(self, method_path, graph_path, res_path):
        super().__init__(method_path, graph_path, res_path)

    def getcomplexes(self, cmd_res):
        with open(self.res_path, 'w') as f:
            for index, item in enumerate(cmd_res):
                if index % 2 == 0:
                    f.write(item)


class clique_method(baseMethod):
    def __init__(self, method_path, graph_path, res_path):
        super().__init__(method_path, graph_path, res_path)

    def getcomplexes(self, cmd_res):
        with open(self.res_path, 'w') as f:
            for item in cmd_res[1:]:
                f.write(item)


class mcode_method(baseMethod):
    def __init__(self, method_path, graph_path, res_path):
        super().__init__(method_path, graph_path, res_path)

    def getcomplexes(self, cmd_res):
        res = []
        start = False
        for item in cmd_res:
            if start:
                if len(item) and "\r" not in item:
                    res.append(item)
            if item == "molecular complex prediction\r\n":
                start = True
        with open(self.res_path, 'w') as f:
            for item in res:
                f.write(item)


def get_method(name):
    if name == "ipca":
        return ipca_method
    if name == "dpclus":
        return dpclus_method
    if name == "clique":
        return clique_method
    if name == "mcode":
        return mcode_method
    return None


if __name__ == "__main__":
    random.seed(666)
    nodeWithFeat_path = "Data/Yeast/embedding/dip_node"
    edgeWithFeat_path = "Data/Yeast/embedding/dip_edge"
    edge_path = "Data/Yeast/embedding/dip_edge_nofeat"
    model_path = "Model/saved_models_gcnbase_11_28_19_56/30.pt"
    bench_path = "Data/Yeast/bench/CYC2008"

    method_name = "mcode"
    methodor = get_method(method_name)
    method_path = "Utils/ReferMethods/{}.py".format(method_name)
    method_expand_path = "Utils/ReferMethods/{}_expand.py".format(method_name)
    complexes_path = "Data/Yeast/compare/dip_{}".format(method_name)
    complexes_expand_path = complexes_path+"_expand"
    subgraphs_path = complexes_path+"_graphs"
    subgraphs_expand_path = complexes_expand_path+"_graphs"

    normal_datas = methodor(method_path, edge_path,
                            complexes_path).main(reload=True)
    expand_candi_datas = methodor(method_expand_path, edge_path,
                                  complexes_expand_path).main(reload=True)
    expand_candi_graphs = data.second_stage(
        nodeWithFeat_path, edgeWithFeat_path, complexes_expand_path, subgraphs_expand_path, reload=True, direct=False)
    expand_candi_graphs = [[item.graph, item.feat]
                           for item in expand_candi_graphs]
    nodefeatsize = 420
    edgefeatsize = 10
    graphfeatsize = 10
    batchsize = 10
    base_model = graph_classify.GCNBASEModel(
        nodefeatsize=nodefeatsize,
        edgefeatsize=edgefeatsize,
        graphfeatsize=graphfeatsize,
        hidden_size=16,
        gcn_layers=2,
        classnum=3
    )
    base_model.load_state_dict(torch.load(model_path))
    res = model.select(base_model, expand_candi_graphs, 0.4)
    expand_datas = []
    for index, val in enumerate(res):
        if val:
            expand_datas.append(expand_candi_datas[index])

    bench_datas = data.read_bench(bench_path)
    f1computor = metrix.ClusterQualityF1(
        bench_datas, expand_datas, metrix.OLAffinity, 0.25)
    res = f1computor.score()
    print(res)
