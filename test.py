        self.random_id = random
        self.random_index = self.id_2_index(self.random_id)
        self.random_graph = self.extract(self.dgl_graph, self.random_index)