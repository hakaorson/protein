import time

import requests
from bs4 import BeautifulSoup


def download_cyc2008_single(id):
    url = 'http://wodaklab.org/cyc2008/complex/show/{}'.format(id)
    response = requests.get(url)
    result = set()
    soup = BeautifulSoup(response.text, features='lxml')
    items = soup.find_all('table')[1].find('tbody').find_all('tr')
    for item in items:
        result.add(item.find('td').text)
    return result


def anasys_dip_item(item):
    splited = item.split('|')
    result = {'dip': None, 'refseq': None,  'ukp': None}
    result['dip'] = splited[0]
    if len(splited) == 3:
        result['refseq'] = splited[1].split(':')[-1]
        result['ukp'] = splited[2].split(':')[-1]
    return result


def get_infos(path):
    resPairs = []
    resSet = set()
    with open(path, 'r') as f:
        next(f)
        for line in f:
            list_info = line.strip().split('\t')
            l_info = anasys_dip_item(list_info[0])
            r_info = anasys_dip_item(list_info[1])
            if l_info['dip'] is not None:
                resSet.add(l_info['dip'])
            if r_info['dip'] is not None:
                resSet.add(r_info['dip'])
            resPairs.append([l_info['dip'], r_info['dip']])
    return resPairs, resSet


def save_set(sets, path):
    with open(path, 'w')as f:
        for item in sets:
            f.write(item+' ')

# 网站限制，暂时不可用
# def items_to_download(sets, path):
#     with open(path, 'w')as f:
#         for item in sets:
#             url = 'https://dip.doe-mbi.ucla.edu/dip/DIPview.cgi?PK={}'.format(
#                 item.replace('DIP-', '').replace('N', ''))
#             response = requests.get(
#                 "https://dip.doe-mbi.ucla.edu/dip/DIPview.cgi?PK=1574", verify=False)
#             soup = BeautifulSoup(response.text, features='html')
#             result = soup.find_all('table')[1].find('tbody').find_all('tr')
#             f.write(item, ' ', result)
#             time.sleep(1)
#             f.write('\n')


if __name__ == "__main__":
    path = r'D:\code\gao_complex\Data\protein\source\network\dip_22977.txt'
    np_path = r'D:\code\gao_complex\Data\protein\source\network\dip_22977_NP'
    resPairs, resSets = get_infos(path)
    save_set(resSets, np_path)
    # items_to_download(resSets, np_path)
