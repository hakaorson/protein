# from Check import process
# # python D:\code\gao_complex\GraphCut\other_methods\coach.py D:\code\gao_complex\Data\dip2.str.tab
# result = process.main_process('Data/CYC2008_408', 'Data/cut/dip2_coach')
# pass
from Data import data
from Model import graph_classify
import torch
import pickle

datas = data.main(reload=False)
nodefeatsize=420
edge
model = graph_classify.SimpleModel(
    graphfeat_size=1, basefeat_size=10, hidden_size=16, gcn_layers=4, classnum=3)

cross_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for i in range(10000):
    epoch_loss = 0
    data_gener = base_data.BatchGenerator(datas, 5)
    for batch_data in data_gener:
        batch_loss = 0
        for data in batch_data:
            predict = model(data.dgl_graph, data.base_feature)
            primary_node_target = torch.tensor(
                data.label, dtype=torch.long).reshape(-1)
            loss = cross_loss(predict, primary_node_target)
            batch_loss += loss
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print('batch loss:', batch_loss.detach().numpy())
    print('epoch loss:', epoch_loss.detach().numpy())
pass
