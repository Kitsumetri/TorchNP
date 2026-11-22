import numpy as np
from numba import njit

"""
Ориентировался на логику из мсе, чтобы было точное сравнение реализаций
https://docs.pytorch.org/docs/2.15/generated/torch.nn.MSELoss.html
"""

# Хоть njit мб и не даёт большого выигрыша, но лишним не будет
@njit(cache=True)
def mse_loss(y_pred: np.ndarray, y_true: np.ndarray):
    # считаем L (лосс) и dL / dY
    # в торче дефолт torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    # поэтому считаем среднее
    
    # на всякий случай
    assert y_pred.shape == y_true.shape
    
    N, C = y_pred.shape
    # C - классы, N - кол-во сэмплов
    diff = y_pred - y_true
    
    loss = np.sum(diff ** 2) / (N * C)
    
    # Y* = y_pred, Y = y_true
    # D = Y* - Y
    # L = <D, D> * 1 / (NC)
    
    # dL = 1 / (NC) * 2 * <D, dD> = 1 / (NC) * 2 <D, dY*>  = <2 / (NC) * D, dY*>
    
    # => dL / dY* = 2 / (NC) * D
    dY = (2.0 / (N * C)) * diff
    
    return float(loss), dY
