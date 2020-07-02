class GeneExpress():
    def __init__(self, path):
        self.json = self.read_from_file(path)

    def read_from_file(self, path):
        result = {}
        with open(path, 'r')as f:
            head = next(f)
            for line in f:
                path
        return result
