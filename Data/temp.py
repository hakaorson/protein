import pandas as pd


def getallids(pathlist):
    res = set()
    for path in pathlist:
        with open(path) as f:
            for line in f:
                linelist = line.strip().split('\t')
                for singleid in linelist:
                    res.add(singleid)
    return res


def save(datas, path):
    with open(path, 'w') as f:
        for data in datas:
            f.write(data+'\n')


if __name__ == "__main__":
    node_path = "embedding/dip_node"
    edge_path="embedding/dip_edge"
    bench_path=
