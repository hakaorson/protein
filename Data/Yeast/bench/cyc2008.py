# 根据cyc2008官方数据集，在uniprot网站上找到其对应的编码
def readmap(path):
    res = {}
    with open(path) as f:
        next(f)
        for line in f:
            linelist = line.strip().split('\t')
            keys = linelist[0].split(',')
            value = linelist[1]
            for key in keys:
                assert(key not in res.keys())
                res[key] = value
    return res


# 读取cyc2008数据集
def readcomplex(path):
    res = []
    with open(path) as f:
        next(f)
        pre_complexname = ""
        for line in f:
            linelist = line.strip().split('\t')
            now_complexname = linelist[2]
            now_protein = linelist[0]
            if now_complexname == pre_complexname:
                res[-1].append(now_protein)
            else:
                res.append([now_protein])
                pre_complexname = now_complexname
    return res


def reid_genename_uniprot(complexes_genename, maps):
    res = []
    for comp in complexes_genename:
        res.append([])
        for prot in comp:
            res[-1].append(maps[prot])
    return res


def save(datas, path):
    with open(path, 'w') as f:
        for data in datas:
            singleline = "\t".join(data)+'\n'
            f.write(singleline)


if __name__ == "__main__":
    cyccomlex = r'D:\code\gao_complex\Data\protein\source\bench\CYC2008_complex'
    cycmap = r"D:\code\gao_complex\Data\protein\source\bench\CYC2008_map"
    cycresult = r"D:\code\gao_complex\Data\protein\source\bench\CYC2008"
    maps = readmap(cycmap)
    complexes_genename = readcomplex(cyccomlex)
    complexes_uniprot = reid_genename_uniprot(complexes_genename, maps)
    save(complexes_uniprot, cycresult)
