import pandas as pd


def save(datas, path):
    with open(path, 'w') as f:
        for data in datas:
            f.write(data+'\n')


def read_graph(graph_path):
    nodes, edges = set(), set()
    with open(graph_path) as f:
        for line in f:
            linelist = tuple(line.strip().split('\t'))
            edges.add(linelist)
            for singleid in linelist:
                nodes.add(singleid)
    return nodes, edges


if __name__ == "__main__":
    dippath = r'D:\code\gao_complex\Data\Yeast\network\dip'
    nodes, edges = read_graph(dippath)
    save(nodes, r'D:\code\gao_complex\Data\Yeast\embedding\uniprotkb_ids')
    source_embed = r'D:\code\gao_complex\Data\Yeast\embedding\uniprotkb_datas'
    processed_embed = r'D:\code\gao_complex\Data\Yeast\embedding\embed'
