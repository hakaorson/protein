import time

import requests
from bs4 import BeautifulSoup


def anasys_dip_item(item):
    splited = item.split('|')
    result = {'dip': None, 'refseq': None,  'uniprotkb': None}
    result['dip'] = splited[0]
    for item in splited[1:]:
        name, data = item.split(':')
        if name == 'refseq':
            result['refseq'] = data
        if name == 'uniprotkb':
            result['uniprotkb'] = data
    return result


def get_infos(path):
    resPairs, resIds = set(), set()
    with open(path, 'r') as f:
        next(f)
        nomatchnum, selfloopnum, repeatnum = 0, 0, 0
        for line in f:
            list_info = line.strip().split('\t')
            l_info = anasys_dip_item(list_info[0])
            r_info = anasys_dip_item(list_info[1])
            if l_info['uniprotkb'] is None or r_info['uniprotkb'] is None:
                print("No match:", l_info['uniprotkb'], r_info['uniprotkb'])
                nomatchnum += 1
                continue
            if l_info['uniprotkb'] == r_info['uniprotkb']:
                print("Loop:", l_info['uniprotkb'], r_info['uniprotkb'])
                selfloopnum += 1
                continue
            if (l_info['uniprotkb'], r_info['uniprotkb']) in resPairs or (r_info['uniprotkb'], l_info['uniprotkb']) in resPairs:
                print("Reapeat:", l_info['uniprotkb'], r_info['uniprotkb'])
                repeatnum += 1
                continue
            resPairs.add((l_info['uniprotkb'], r_info['uniprotkb']))
            resIds.add(l_info['uniprotkb'])
            resIds.add(r_info['uniprotkb'])
        print("Not match:{},Loop:{},Reapeat:{},Lefted:{}".format(
            nomatchnum, selfloopnum, repeatnum, len(resPairs)))
    return resPairs, resIds


def read_id(path):
    nodes = set()
    with open(path) as f:
        for line in f:
            linelist = line.strip().split('\t')
            for singleid in linelist:
                nodes.add(singleid)
    return nodes


def read_map(path):
    nodes = set()
    with open(path) as f:
        for line in f:
            linelist = line.strip().split('\t')
            nodes.add(linelist[0])
    return nodes


def save_set(datas, path):
    with open(path, 'w')as f:
        for item in datas:
            if not isinstance(item, str):
                single_line = '\t'.join(item)+'\n'
            else:
                single_line = item+'\n'
            f.write(single_line)


if __name__ == "__main__":
    source_path = r'D:\code\gao_complex\Data\Yeast\network\dip_22977.txt'
    id_path = r'D:\code\gao_complex\Data\Yeast\network\dip_ids'
    res_path = r'D:\code\gao_complex\Data\Yeast\network\dip_22977'
    resPairs, resIds = get_infos(source_path)
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
