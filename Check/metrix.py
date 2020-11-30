import collections
import numpy as np


class ItemAffinity():
    def __init__(self, list_a, list_b):
        self.list_a = list_a
        self.list_b = list_b
        self.set_a = set(self.list_a)
        self.set_b = set(self.list_b)
        self.score = self.score()

    def score(self):
        NotImplemented


class NAAffinity(ItemAffinity):  # neighborhood affinity score
    def __init__(self, list_a, list_b):
        super().__init__(list_a, list_b)

    def score(self):
        set_and = self.set_a & self.set_b
        result = pow(len(set_and), 2)/(len(self.set_a)*len(self.set_b))
        return(result)


class OLAffinity(ItemAffinity):  # overlap score
    def __init__(self, list_a, list_b):
        super().__init__(list_a, list_b)

    def score(self):
        set_and = self.set_a & self.set_b
        result = len(set_and)/(len(self.set_a)*len(self.set_b))
        return(result)


class CoocAffinity(ItemAffinity):  # co-occurrence score,共同蛋白质数量
    def __init__(self, list_a, list_b):
        super().__init__(list_a, list_b)

    def score(self):
        set_and = self.set_a & self.set_b
        result = len(set_and)
        return(result)


class ClusterQuality():
    def __init__(self, cluster_bench, cluster_predict, affinity_method):
        self.cluster_bench = cluster_bench
        self.cluster_predict = cluster_predict
        self.id_map = self.get_id_map()
        self.affinity_method = affinity_method
        self.affinity_matrix = self.get_affinity_matrix()

    def get_id_map(self):
        id_map = collections.defaultdict(int)
        all_item = set([item for cluster in self.cluster_bench for item in cluster]) | \
            set([item for cluster in self.cluster_predict for item in cluster])
        for index, item in enumerate(all_item):
            id_map[item] = index
        return(id_map)

    def get_affinity_matrix(self):
        af_matrix = [[0 for j in range(len(self.cluster_predict))]for i in range(
            len(self.cluster_bench))]
        for i in range(len(self.cluster_bench)):
            for j in range(len(self.cluster_predict)):
                ij_affinity = self.affinity_method(
                    self.cluster_bench[i], self.cluster_predict[j])
                af_matrix[i][j] = ij_affinity.score
        return af_matrix

    def score(self):
        NotImplemented


class ClusterQualityF1(ClusterQuality):
    def __init__(self, cluster_bench, cluster_predict, affinity_method=None, threshold=None):
        self.threshold = threshold
        super().__init__(cluster_bench, cluster_predict, affinity_method)

    def score(self):
        np_matrix = np.array(self.affinity_matrix)
        bool_matrix = np_matrix >= self.threshold
        sum_matrix = np.sum(bool_matrix, 0) > 0  # 这个计算是不是有问题
        result = sum(sum_matrix)/len(sum_matrix)
        return(result)
