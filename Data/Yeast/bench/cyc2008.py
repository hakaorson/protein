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
    uniprots = []
    for comp in complexes_genename:
        uniprots.append([])
        for prot in comp:
            uniprots[-1].append(maps[prot])
    return uniprots


def save(datas, path):
    with open(path, 'w') as f:
        for data in datas:
            singleline = "\t".join(data)+'\n'
            f.write(singleline)


if __name__ == "__main__":
    cyccomlex = 'CYC2008_complex'
    cycmap = "CYC2008_map"
    cycuniprot = "CYC2008"
    cycgenename = "CYC2008_gene"
    maps = readmap(cycmap)
    complexes_genename = readcomplex(cyccomlex)
    complexes_uniprot = reid_genename_uniprot(complexes_genename, maps)
    save(complexes_uniprot, cycuniprot)
    save(complexes_genename, cycgenename)
