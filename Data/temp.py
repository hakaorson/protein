import pandas as pd


def getallids(pathlist):
    res = set()
    for path in pathlist:
        with open(path) as f:
            for line in f:
                linelist = line.strip().split('\t')
                for singleid in linelist:
                    res.add(singleid)
    return res


def save(datas, path):
    with open(path, 'w') as f:
        for data in datas:
            f.write(data+'\n')


if __name__ == "__main__":
    cycpath = r'D:\code\gao_complex\Data\Yeast\bench\CYC2008'
    dippath = r'D:\code\gao_complex\Data\Yeast\network\dip'
    cycids = getallids([cycpath])
    dipids = getallids([dippath])
    cyc_notin_dip = cycids-dipids
    with open(cycpath, 'r') as f:
        res = []
        for line in f:
            linelist = set(line.strip().split('\t'))
            temp = linelist & cyc_notin_dip
            if len(temp):
                continue
            res.append(linelist)
    source_embed = r'D:\code\gao_complex\Data\Yeast\embedding\uniprotkb_datas'
    processed_embed = r'D:\code\gao_complex\Data\Yeast\embedding\embed'
    getdata(source_embed, processed_embed)

    # with open(source_embed, 'r') as fromfile, open(processed_embed, 'w') as tofile:
    #     head = next(fromfile).strip().split('\t')
    #     for line in fromfile:
    #         linelist = line.strip().split('\t')
    #         writestr = '\t'.join([linelist[0], '1'])+'\n'
    #         tofile.write(writestr)
