# 读取cyc2008数据集
def readcomplex(path):
    res_complex, res_id = [], set()
    with open(path) as f:
        next(f)
        pre_complexname = ""
        for line in f:
            linelist = line.strip().split('\t')
            now_complexname = linelist[2]
            now_protein = linelist[0]
            res_id.add(now_protein)
            if now_complexname == pre_complexname:
                res_complex[-1].append(now_protein)
            else:
                res_complex.append([now_protein])
                pre_complexname = now_complexname
    return res_complex, res_id


def save_set(datas, path):
    with open(path, 'w')as f:
        for item in datas:
            if not isinstance(item, str):
                single_line = '\t'.join(item)+'\n'
            else:
                single_line = item+'\n'
            f.write(single_line)


if __name__ == "__main__":
    cyc_comlex = r'D:\code\gao_complex\Data\Yeast\bench\CYC2008_locus_complex'
    cyc_id = r'D:\code\gao_complex\Data\Yeast\bench\CYC2008_locus_id'
    cyc_result = r'D:\code\gao_complex\Data\Yeast\bench\CYC2008'
    complexes_locus, ids_locus = readcomplex(cyc_comlex)
    save_set(complexes_locus, cyc_result)
    save_set(ids_locus, cyc_id)
