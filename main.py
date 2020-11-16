# from Check import process
# # python D:\code\gao_complex\GraphCut\other_methods\coach.py D:\code\gao_complex\Data\dip2.str.tab
# result = process.main_process('Data/CYC2008_408', 'Data/cut/dip2_coach')
# pass
from Data import data
from Model import graph_classify
import torch
import pickle

datas = data.main(reload=False)
nodefeatsize = 420
edgefeatsize = 10
graphfeatsize = 10
batchsize = 20
model = graph_classify.SimpleModel(
    nodefeatsize=nodefeatsize,
    edgefeatsize=edgefeatsize,
    graphfeatsize=graphfeatsize,

    hidden_size=16,
    gcn_layers=2,
    classnum=3
)

cross_loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([10, 1, 1]))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for i in range(10000):
    epoch_loss = 0
    data_gener = data.BatchGenerator(datas, batchsize)
    for batch_data in data_gener:
        batch_loss = 0
        for item in batch_data:
            primary_node_target = torch.tensor(
                item.label, dtype=torch.long).reshape(-1)
            predict = model(item.graph, item.feat)

            loss = cross_loss(predict, primary_node_target)
            batch_loss += loss
            epoch_loss += loss
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # print('batch loss:', batch_loss.detach().numpy())
    print('epoch loss:', epoch_loss.detach().numpy())
pass
