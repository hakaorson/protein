def read_fasta(input):
    with open(input, 'r') as f:
        fasta = {}
        for line in f:
            line = line.strip()
            if line[0] == '>':
                header = line[1:]
            else:
                sequence = line
                fasta[header] = fasta.get(header, '') + sequence
    res = []
    for key in fasta.keys():
        single_str = '>'+key+'\n'+fasta[key]
        res.append(single_str)
    return res


def save_fasta(path, data):
    with open(path, 'w') as f:
        for single_str in data:
            f.write(single_str+'\n')


if __name__ == "__main__":
    path = r"D:\code\gao_complex\Data\Yeast\embedding\blast"
    datas = read_fasta(path+'\dip.fasta')
    begin, fileindex = 0, 0
    while begin < len(datas):
        end = begin+1000
        save_fasta(path+'\dip{}.fasta'.format(fileindex), datas[begin:end])
        fileindex += 1
        begin = end
