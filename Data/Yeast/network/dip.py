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
    resPairs = []
    with open(path, 'r') as f:
        next(f)
        for line in f:
            list_info = line.strip().split('\t')
            l_info = anasys_dip_item(list_info[0])
            r_info = anasys_dip_item(list_info[1])
            if l_info['uniprotkb'] is None or r_info['uniprotkb'] is None:
                print(l_info, r_info)
                continue
            resPairs.append([l_info['uniprotkb'], r_info['uniprotkb']])
    return resPairs


def save_set(datas, path):
    with open(path, 'w')as f:
        for item in datas:
            single_line = '\t'.join(item)+'\n'
            f.write(single_line)


if __name__ == "__main__":
    path = r'D:\code\gao_complex\Data\Yeast\network\dip_22977.txt'
    np_path = r'D:\code\gao_complex\Data\Yeast\network\dip_22977'
    resPairs = get_infos(path)
    save_set(resPairs, np_path)
