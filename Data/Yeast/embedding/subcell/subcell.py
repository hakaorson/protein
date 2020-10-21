def readBioText(path, split, cut):
    res = []
    with open(path, 'r') as f:
        tempinfo = {}
        for index, line in enumerate(f):
            if index < 43 or index >= 6292:
                continue
            line = line.strip()
            if line == split:
                res.append(tempinfo)
                tempinfo = {}
            else:
                datas = line.split(cut)
                if len(datas) == 2:
                    key, val = datas
                    tempinfo[key] = tempinfo.get(key, '')+val
    return res


if __name__ == "__main__":
    datas = readBioText(
        r'D:\code\gao_complex\Data\Yeast\embedding\subcell\subcell.txt', r'//', '   ')
    mapping = dict()
    mapping_path = r'D:\code\gao_complex\Data\Yeast\embedding\subcell\mapping'
    for item in datas:
        key = ''
        key = item['ID'] if (key == '' and 'ID' in item.keys()) else key
        key = item['IT'] if (key == '' and 'IT' in item.keys()) else key
        key = item['IO'] if (key == '' and 'IO' in item.keys()) else key
        key = key.replace('.', '')
        val = item['AC']
        if key != '':
            mapping[key] = val
    with open(mapping_path, 'w')as f:
        for key in mapping.keys():
            singleline = '\t'.join([key, mapping[key]])+'\n'
            f.write(singleline)
