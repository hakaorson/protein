import pandas as pd
import re


def findSubcellWords(str_input):
    str_remove_head = re.sub('SUBCELLULAR LOCATION: ', "", str_input)
    str_remove_bracket = re.sub('{.*}', "", str_remove_head)
    str_remove_note = re.sub('Note=.*', "", str_remove_bracket)
    str_splited = re.split('\.|;|,', str_remove_note)
    result = []
    for single_str in str_splited:
        single_str = single_str.strip().capitalize()
        if single_str:
            result.append(single_str)
    # print(result)
    return result


def save(datas, path):
    with open(path, 'w') as f:
        for data in datas:
            f.write(data+'\n')


# 读取graph
def read_graph(graph_path):
    nodes, edges = set(), set()
    with open(graph_path) as f:
        for line in f:
            linelist = tuple(line.strip().split('\t'))
            edges.add(linelist)
            for singleid in linelist:
                nodes.add(singleid)
    return nodes, edges


def read_uniprotkb(path):
    res = {}
    with open(path, 'r')as f:
        heads = next(f)
        headslist = heads.strip().split('\t')
        enterIndex = headslist.index('Entry')
        seqIndex = headslist.index('Sequence')
        subcellIndex = headslist.index('Subcellular location [CC]')
        goIndex = headslist.index('Gene ontology IDs')
        domainIndex = headslist.index('Cross-reference (Pfam)')
        for line in f:
            linelist = line.split('\t')
            data = {}
            data['seq'] = linelist[seqIndex]
            data['go'] = linelist[goIndex].replace(' ', '').split(';')
            data['subcell'] = findSubcellWords(linelist[subcellIndex])
            data['domain'] = linelist[domainIndex][:-1].split(';')
            res[linelist[enterIndex]] = data
    return res


def compute_edge_feats(edges, nodedatas):
    pass


def compute_node_feats(nodes, nodedatas):
    pass


if __name__ == "__main__":
    dippath = r'D:\code\gao_complex\Data\Yeast\network\dip'
    nodes, edges = read_graph(dippath)
    save(nodes, r'D:\code\gao_complex\Data\Yeast\embedding\uniprotkb_ids')
    uniprotkb_path = r'D:\code\gao_complex\Data\Yeast\embedding\uniprot-yourlist_M20201020A94466D2655679D1FD8953E075198DA80588E6L.tab'
    uniprotkb_datas = read_uniprotkb(uniprotkb_path)

    compute_edge_feats(edges, uniprotkb_datas)
    compute_node_feats(nodes, uniprotkb_datas)

    processed_embed = r'D:\code\gao_complex\Data\Yeast\embedding\embed'
