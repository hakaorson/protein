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


def getdata(uni_path, to_path):
    df = pd.read_table(uni_path)
    pass


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
    source_embed = r'D:\code\gao_complex\Data\Yeast\embedding\uniprotkb_datas'
    processed_embed = r'D:\code\gao_complex\Data\Yeast\embedding\embed'
    getdata(source_embed, processed_embed)

    # with open(source_embed, 'r') as fromfile, open(processed_embed, 'w') as tofile:
    #     head = next(fromfile).strip().split('\t')
    #     for line in fromfile:
    #         linelist = line.strip().split('\t')
    #         writestr = '\t'.join([linelist[0], '1'])+'\n'
    #         tofile.write(writestr)
