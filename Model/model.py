from Model import graph_classify
import torch
from Data import data
import pickle
import dgl
import os


def getmodel(nodefeatsize, edgefeatsize, graphfeatsize):
    model = graph_classify.SimpleModel(
        nodefeatsize=nodefeatsize,
        edgefeatsize=edgefeatsize,
        graphfeatsize=graphfeatsize,

        hidden_size=16,
        gcn_layers=2,
        classnum=3
    )
    return model


def collate(samples):
    graphs, feats, labels = map(list, zip(*samples))
    batch_graph = dgl.batch(graphs)
    return batch_graph, feats, torch.tensor(labels)


def train(model, datas, batchsize, path):
    os.makedirs(path, exist_ok=True)
    cross_loss = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor([1, 1, 1]))  # 这苦有问题
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for i in range(10000):
        epoch_loss = 0
        # data_loader = torch.utils.data.DataLoader(
        #     datas, batch_size=batchsize, shuffle=True, collate_fn=collate)
        # for iter, (graphs, feats, label) in enumerate(data_loader):
        #     prediction = model(graphs, feats)
        #     loss = cross_loss(prediction, label)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     epoch_loss += loss.detach().item()  # 每一个批次的损失
        data_gener = data.BatchGenerator(datas, batchsize)
        for batch_data in data_gener:
            batch_loss = 0
            for item in batch_data:
                primary_node_target = torch.tensor(
                    item[2], dtype=torch.long).reshape(-1)
                predict = model(item[0], item[1])

                loss = cross_loss(predict, primary_node_target)
                batch_loss += loss
                epoch_loss += loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # print('batch loss:', batch_loss.detach().numpy())
        if i != 0 and i % 10 == 0:
            torch.save(model.state_dict(), path+'/{}.pt'.format(i))
        print('epoch {} loss:'.format(i), epoch_loss.detach().numpy())


def test(model, datas):
    labels = []
    predicts = []
    for item in datas:
        labels.append(item[2])
        pred = model(item[0], item[1])
        pred = list(pred[0].detach().numpy())
        predindex = pred.index(max(pred))
        predicts.append(predindex)
        pass
    static_recall = [[0, 0], [0, 0], [0, 0]]
    static_precision = [[0, 0], [0, 0], [0, 0]]
    for index in range(len(datas)):
        truelabel = labels[index]
        predictlabel = predicts[index]
        static_recall[truelabel][1] += 1
        static_precision[predictlabel][1] += 1
        if truelabel == predictlabel:
            static_recall[truelabel][0] += 1
            static_precision[predictlabel][0] += 1
    for index in range(len(static_recall)):
        recallnum = static_recall[index][0]/static_recall[index][1]
        precinum = static_precision[index][0]/static_precision[index][1]
        f1num = 2*recallnum*precinum/(recallnum+precinum)
        print("recall {},prec {},f1 {}".format(recallnum, precinum, f1num))


if __name__ == "__main__":
    pass
