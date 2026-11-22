"""
Старался соблюдать конвенции торч интерфейсов оптимизаторов, поэтому всё через zero_grad и step
https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
"""

class SGDOptimizerNP:
    def __init__(self, params: list, lr: float):
        self.params = params
        self.lr = lr
        
    def zero_grad(self):
        # обнуляем все градиенты
        for p in self.params:
            p.dW[...] = 0.0 # по весам
            p.db[...] = 0.0 # по байесам
            
    def step(self):
        # шаг градиента
        # w_{t+i} = w_t - lr * grad(L(w_t))
        for p in self.params:
            p.W -= self.lr * p.dW # по весам
            p.b -= self.lr * p.db # по байесам
