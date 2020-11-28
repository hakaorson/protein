from Data.Yeast import data
from Model import graph_classify
from Model import model
import random
import torch
import time
import pickle

if __name__ == "__main__":
    random.seed(666)
    node_path = "Data/Yeast/embedding/dip_node"
    edge_path = "Data/Yeast/embedding/dip_edge"
    model_path = "Model/saved_models_gcnbase_11_28_19_56/30.pt"
    thred = 0.5
    normal_datas = data.read_bench(normal_path)  # 通过获取子进程输出
    expand_candi_datas = data.read_bench(expand_path)
    expand_candi_graphs = data.second_stage(
        node_path, edge_path, expand_path, direct=False)
    expand_candi_graphs = [[item.graph, item.feat]
                           for item in expand_candi_graphs]

    nodefeatsize = 420
    edgefeatsize = 10
    graphfeatsize = 10
    batchsize = 10
    base_model = graph_classify.SimpleModel(
        nodefeatsize=nodefeatsize,
        edgefeatsize=edgefeatsize,
        graphfeatsize=graphfeatsize,
        hidden_size=16,
        gcn_layers=2,
        classnum=3
    )
    base_model.load_state_dict(torch.load(model_path))
    res = model.select(base_model, expand_candi_graphs, thred)
    expand_datas = []
    for index, val in enumerate(res):
        if val:
            expand_datas.append(expand_candi_datas[index])
    # 到目前为止准备好了normal和expand
