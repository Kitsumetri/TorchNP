"""
class MLPClassifier(torch.nn.Module):
    def __init__(self, in_features: int = 784, hidden: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden)
        self.fc2 = torch.nn.Linear(hidden, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""

# Ссылаюсь на свою же реализацию на торче, чтобы построить похожий интерфейс

from abc import ABC, abstractmethod
import numpy as np
from numba import njit


# Как мне кажется единственный норм опитимизации - это njit на relu
# По поводу матричного умножения в FC - хз
# В нампае сидят не дураки, там умные Blass или Atlass алгоритмы на С, кто-то говорит, что уже лучшие, кто-то нет.
# Параллелить в обучении матмул через numba - шанс заработать оверхеад на гонке данных
# fastmath тоже может уронить стабильность обучения, особенно когда всё в fp32
# Поэтому решил отказаться от numba оптимизонов для Linear слоёв.
# https://stackoverflow.com/questions/10442365/why-is-matrix-multiplication-faster-with-numpy-than-with-ctypes-in-python

@njit(cache=True)
def _relu_forward(x):
    n, m = x.shape
    out = np.empty((n, m), dtype=x.dtype)
    for i in range(n):
        for j in range(m):
            v = x[i, j]
            out[i, j] = v if v > 0.0 else 0.0
    return out


@njit(cache=True)
def _relu_backward(dY, x):
    n, m = dY.shape
    out = np.empty((n, m), dtype=dY.dtype)
    for i in range(n):
        for j in range(m):
            out[i, j] = dY[i, j] if x[i, j] > 0.0 else 0.0
    return out

class ModuleNP(ABC):
    # Базовый класс для всех модулей, костыльный torch.nn.Module
    def __init__(self):
        pass
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def backward(self, dY: np.ndarray) -> np.ndarray:
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class LinearNP(ModuleNP):
    # линейный полносвязный слой
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        """
        Инициализация весов и bias - как torch.nn.Linear по дефолту
        Kaiming Uniform
        W_ij ~ U(-1/sqrt(d_in), +1/sqrt(d_in))
        b_j  ~ U(-1/sqrt(d_in), +1/sqrt(d_in))
    
        d_in = in_features
        
        Все эти финты нужны, чтобы поставить нампай и торч в одинаковые условия
        Проводил эксперименты, где инитил веса через "He" и просто нулями - результат хуже торча был в обучении
        """
        bound = 1.0 / np.sqrt(in_features)
        self.W = np.random.uniform(-bound, bound, size=(in_features, out_features)).astype(np.float32)
        self.b = np.random.uniform(-bound, bound, size=(out_features,)).astype(np.float32)
        
        self.cache_X = None # кэширование X для backward
        self.dW = np.zeros_like(self.W, dtype=np.float32)
        self.db = np.zeros_like(self.b, dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Y = X @ W + b
        self.cache_X = np.copy(x)
        return x @ self.W + self.b
    
    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Forward: Y = X @ W + b
        Дифференциал fowrard: dY = dX @ W + X @ dW + I_N @ db^T
        I_N — столбец единиц (N x 1)
        
        Дифференциал loss, для удобства G = dL/dY: dL = <G, dY> = trace(G^T @ dY)
        
        Подставляем dY и раскрываем по свойству следа (trace(AB) = trace(BA)):
        dL = trace(G^T @ dX @ W) + trace(G^T @ X @ dW) + trace(G^T @ 1_N @ db^T)
        dL = <G @ W^T, dX> + <X^T @ G, dW> + <G^T @ 1_N, db>
        
        Сравнивая с dL = <dL/dX, dX> + <dL/dW, dW> + <dL/db, db>:
        dL/dX = G @ W^T
        dL/dW = X^T @ G
        dL/db = G^T @ I_N = sum_n G[n, :]
        """
        self.dW = self.cache_X.T @ dY
        self.db = dY.sum(axis=0)
        
        dX = dY @ self.W.T
        return dX


class ReLUNP(ModuleNP):
    # функция акитивации ReLU
    def __init__(self):
        super().__init__()
        self.cache_x = None # кэш X для backward
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # y_i = max(0, x_i)
        self.cache_x = np.copy(x)
        
        # Раньше было так
        # return np.maximum(x, 0)
        
        # Теперь так
        return _relu_forward(x)
    
    def backward(self, dY: np.ndarray) -> np.ndarray:        
        # dL / dX = (dL / dY) * (dY / dX)
        
        # ReLU'(x) = 
        # 1, if x > 0
        # 0, if x < 0
        
        # Поэтому dY / dx = np.where(x > 0. 1. 0)
        
        # dL / dX = (dL / dY) * np.where(x > 0. 1. 0)
        # dX = dY * np.where(x > 0. 1. 0)
        
        # Раньше было так
        # return dY * np.where(self.cache_x > 0, 1.0, 0.0) 
        
        # Теперь так
        return _relu_backward(dY, self.cache_x)
    

class MLPClassifierNP(ModuleNP):
    """Собрал похожий интерфейс, как у торчевского"""
    def __init__(self, in_features: int = 784, hidden: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = LinearNP(in_features, hidden)
        self.fc2 = LinearNP(hidden, num_classes)
        self.relu = ReLUNP()
    
    def forward(self, x: np.ndarray):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def backward(self, dY: np.ndarray):
        dY = self.fc2.backward(dY)
        dY = self.relu.backward(dY)
        dY = self.fc1.backward(dY)
        return dY
