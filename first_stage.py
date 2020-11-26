# from Check import process
# # python D:/code/gao_complex/GraphCut/other_methods/coach.py D:/code/gao_complex/Data/dip2.str.tab
# result = process.main_process('Data/CYC2008_408', 'Data/cut/dip2_coach')
# pass
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
    postive_path = "Data/Yeast/bench/CYC2008"
    middle_path = "Data/Yeast/bench/dip_coach"
    save_path = "Data/Yeast/first_stage"
    datas = data.first_stage(node_path, edge_path,
                             postive_path, middle_path, save_path, reload=True)
    datas = [[item.graph, item.feat, item.label] for item in datas]
    random.shuffle(datas)
    size = len(datas)
    cut1, cut2 = int(0.7*size), int(0.85*size)
    traindatas, testdatas = datas[:cut1], datas[cut1:]

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
    model_path = "Model/saved_models_{}".format(
        time.strftime('%m_%d_%H_%M', time.localtime()))
    # model.train(base_model, traindatas, batchsize, model_path)

    base_model.load_state_dict(torch.load(
        "Model/saved_models_11_18_10_12/90.pt"))
    model.test(base_model, testdatas)
