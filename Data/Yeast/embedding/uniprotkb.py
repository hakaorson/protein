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
    target_to_findid = [r'D:\code\gao_complex\Data\Yeast\bench\CYC2008',
                        r'D:\code\gao_complex\Data\Yeast\network\dip_22977'
                        ]
    # ids = getallids(target_to_findid)
    # savepath = r'D:\code\gao_complex\Data\Yeast\embedding\uniprokb_ids'
    # save(ids, savepath)
    embedpath = r'D:\code\gao_complex\Data\Yeast\embedding\embed'
    with open(embedpath, 'r') as f:
        for line in f:
            linelist = line.strip().split('\t')
            pass
