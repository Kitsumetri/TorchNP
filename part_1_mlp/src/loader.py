import numpy as np

class DataLoaderNP:
    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 batch_size: int = 128, shuffle: bool = False, seed: int = 42):
        assert X.shape[0] == Y.shape[0]
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.n = X.shape[0]

    def __len__(self):
        # число батчей
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = np.arange(self.n)
        if self.shuffle:
            self.rng.shuffle(idx)
        for i in range(0, self.n, self.batch_size):
            b = idx[i:i + self.batch_size]
            yield self.X[b], self.Y[b]
