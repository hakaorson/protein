import os
if __name__ == "__main__":
    names = os.listdir("POSSUM")
    res = []
    for name in names:
        with open("POSSUM/"+name, 'r') as f:
            next(f)
            datas = list(map(float, next(f).split(',')))
            short_datas = list(map(lambda num: "{:.2f}".format(num), datas))
            strings = name[:6]+'\t'+'\t'.join(short_datas)+'\n'
            res.append(strings)
    with open('POSSUM_DATA', 'w') as f:
        for item in res:
            f.write(item)
