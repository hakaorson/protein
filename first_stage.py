from Data.Yeast import data
from Model import graph_classify
from Model import model
import random
import torch
import time
import pickle

if __name__ == "__main__":
    random.seed(666)
    nodeWithFeat_path = "Data/Yeast/embedding/dip_node"
    edgeWithFeat_path = "Data/Yeast/embedding/dip_edge"
    postive_path = "Data/Yeast/bench/CYC2008"
    middle_path = "Data/Yeast/compare/dip_coach"
    save_path = "Data/Yeast//first_stage_data"
    datas = data.first_stage(nodeWithFeat_path, edgeWithFeat_path,
                             postive_path, middle_path, save_path, reload=True, direct=False)
    # data.analisys_data(datas)  # 统计一下数据的信息
    datas = [[item.graph, item.feat, item.label] for item in datas]
    random.shuffle(datas)
    size = len(datas)
    cut1, cut2 = int(0.7*size), int(0.85*size)
    traindatas, valdatas, testdatas = datas[:
                                            cut1], datas[cut1:cut2], datas[cut2:]

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
    model_path = "Model/saved_models/{}_{}".format(base_model.name,
                                                   time.strftime('%m_%d_%H_%M', time.localtime()))
    default_epoch = 50

    model.train(base_model, traindatas, valdatas,
                batchsize, model_path, default_epoch)
    base_model.load_state_dict(torch.load(
        model_path+'/{}.pt'.format(default_epoch)))
    # base_model.load_state_dict(torch.load(
    #     "Model/saved_models_base_11_28_19_39/10.pt"))
    model.test(base_model, testdatas)
