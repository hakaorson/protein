from Data import base_graph
from Check import metrix
import random


class ExtractGraphs():
    def __init__(self, bench, compare, random):
        self.bench = bench
        self.compare = compare
        self.random = random
        self.add_label()
        self.all_data = self.compare+self.random

    def add_label(self):
        for index in range(len(self.random)):
            self.random[index].label = 0
        bench_list = [item.comple for item in self.bench]
        for index in range(len(self.compare)):
            max_score = 0
            for bench in bench_list:
                score = metrix.NAAffinity(
                    self.compare[index].comple, bench).score
                max_score = max(max_score, score)
            if max_score > 0.25:
                self.compare[index].label = 2
            else:
                self.compare[index].label = 1


class BatchGenerator():
    def __init__(self, data, batch_size):
        self.data = data
        random.shuffle(self.data)
        self.batch_size = batch_size if batch_size != -1 else len(self.data)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.data[self.index+self.batch_size-1]  # 用于检查是否越界
            b_data = self.data[self.index:self.index+self.batch_size]
        except IndexError:
            raise StopIteration()
        self.index += self.batch_size
        return b_data


def main():
    dgl_data = base_graph.main()
    data_to_train = ExtractGraphs(
        dgl_data.all_bench, dgl_data.all_compare, dgl_data.all_random)
    data_generator = BatchGenerator(data_to_train.all_data, 8)
    return data_generator


if __name__ == "__main__":
    main()
