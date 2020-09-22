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


def download_cyc2008_all(path):
    with open(path, 'w')as f:
        for id in range(408):
            temp_info = download_cyc2008_single(id+1)
            time.sleep(1)
            f.write(temp_info)


def prepare_cyc2008(from_path, to_path):
    with open(from_path, 'r') as fromfile, open(to_path, 'w') as tofile:
        for line in fromfile:
            temp = line.strip().replace(
                '}', '').replace('{', '').replace(',', '').replace('\'', '')
            temp2 = temp.split(' ')
            result = temp2[3:]
            newline = ' '.join(result)+'\n'
            tofile.write(newline)
def 

if __name__ == "__main__":
    # 处理cyc2008
    from_path = r'Data\protein\source\bench\CYC2008_408.txt'
    to_path = r"Data\protein\bench\CYC2008_408"
    download_cyc2008_all(from_path)
    prepare_cyc2008(from_path, to_path)
