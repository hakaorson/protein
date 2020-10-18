import time

import requests
from bs4 import BeautifulSoup


def read_graph(graph_path):
    nodes, edges = set(), set()
    with open(graph_path) as f:
        for line in f:
            linelist = tuple(line.strip().split('\t'))
            edges.add(linelist)
            for singleid in linelist:
                nodes.add(singleid)
    return nodes, edges


def save_set(datas, path):
    with open(path, 'w')as f:
        for item in datas:
            if not isinstance(item, str):
                single_line = '\t'.join(item)+'\n'
            else:
                single_line = item+'\n'
            f.write(single_line)


if __name__ == "__main__":
    source_path = r'D:\code\gao_complex\Data\Yeast\network\dip_17201'
    id_path = r'D:\code\gao_complex\Data\Yeast\network\dip_17201_locus_id'
    res_path = r'D:\code\gao_complex\Data\Yeast\network\dip_22977'
    resIds, resPairs = read_graph(source_path)
    save_set(resIds, id_path)
    save_set(resPairs, res_path)

    # 根据dip_ids取uni上下载map的数据...

    map_path = r'D:\code\gao_complex\Data\Yeast\network\dip_id_mapped'
    res_path_mapped = r'D:\code\gao_complex\Data\Yeast\network\dip'
    mapped_nodes = read_map(map_path)
    resPairsMapped = set()
    notYeastNum = 0
    for item in resPairs:
        if item[0] in mapped_nodes and item[1] in mapped_nodes:
            resPairsMapped.add(item)
        else:
            print('Not yeast:', item)
            notYeastNum += 1
    print('Not yeast num:', notYeastNum)
    save_set(resPairsMapped, res_path_mapped)
