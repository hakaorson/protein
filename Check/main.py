from Check import metrix


def get_cluster_info(path):
    with open(path) as file:
        result = []
        for line in file:
            data = list(item.upper() for item in line.split())
            result.append(data)
    return(result)


def main_process(bench_path, predict_path):
    bench_complexes = get_cluster_info(bench_path)
    predict_complexes = get_cluster_info(predict_path)
    return metrix.RecallQuality(bench_complexes, predict_complexes,
                                metrix.NAAffinity, 0.25)
