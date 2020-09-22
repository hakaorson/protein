
def prepare_gene_expression(from_path, to_path):
    result = {}
    with open(from_path, 'r')as f:
        head = next(f)
        genes = head.split(',')
        for index, data in enumerate(f):
            result[genes[index]] = data.strip().split(',')
    with open(to_path, 'w')as f:
        for key in result.keys():
            line = key+' '+' '.join(result[key])+'\n'
            f.write(line)


if __name__ == "__main__":
    from_path = r'D:\code\gao_complex\Data\protein\source\embedding\gene expression data.soft'
    to_path = r'D:\code\gao_complex\Data\protein\embedding\node\gene_expression'
    prepare_gene_expression(from_path, to_path)
