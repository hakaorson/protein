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
    print(id, ' ', result)
    return result


def download_cyc2008_all():
    all_result = []
    for id in range(408):
        all_result.append(download_cyc2008_single(id+1))
        time.sleep(1)
    return all_result


def read_cyc():
    with open('D:\\code\\gao_complex\\Data\\source\\cyc2008_408.txt', 'r') as fromfile, open('D:\\code\\gao_complex\\Data\\CYC2008_408', 'w') as tofile:
        for line in fromfile:
            temp = line.strip().replace(
                '}', '').replace('{', '').replace(',', '').replace('\'', '')
            temp2 = temp.split(' ')
            result = temp2[3:]
            newline = ' '.join(result)+'\n'
            tofile.write(newline)


read_cyc()
