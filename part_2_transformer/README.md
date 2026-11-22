# TransformerEncoderLayer на NumPy: forward и backward без autograd

Реализация одного слоя трансформера, эквивалентного
`torch.nn.TransformerEncoderLayer`, на чистых `numpy.ndarray`.

В проекте **нет autograd**, нет класса `Tensor` и нет графа вычислений.
`forward` сохраняет промежуточные значения, а `backward` вручную применяет
chain rule, накапливает градиенты параметров и возвращает градиент по входу.
Все формулы выведены в docstring-ах:
[`ops.py`](src/nptorch/nn/ops.py) — отдельные операции,
[`transformer.py`](src/nptorch/nn/transformer.py) — attention, FFN, layer norm
и полная цепочка обратного прохода слоя.

## Что реализовано

- `TransformerEncoderLayer` с параметрами, совпадающими по именам и форме с
  `torch.nn.TransformerEncoderLayer`:
  `self_attn.in_proj_weight`, `self_attn.out_proj.weight`, `linear1.weight`,
  `norm1.weight` и т.д.
- Режимы `post-norm` (по умолчанию) и `norm_first=True` (pre-norm).
- Активации `relu` и точный `gelu` (через `erf`).
- `bias=True/False`, `batch_first=True/False`, `layer_norm_eps`, `dropout`.
- Вход `(L, N, E)`, `(N, L, E)` при `batch_first=True` или unbatched `(L, E)`.
- `src_mask` (bool или аддитивная float-маска), `src_key_padding_mask`,
  `is_causal` с семантикой PyTorch MHA.
- `state_dict()` / `load_state_dict()` с torch-ключами: веса можно копировать
  из torch-слоя и обратно.
- `zero_grad()`, `train()`, `eval()`, `params`, `grads`, `named_parameters()`.
- Инициализация повторяет torch: Xavier для `in_proj_weight`, Kaiming для
  линейных весов и `out_proj`, `U(-1/√fan_in, 1/√fan_in)` для bias внутри
  FFN, нули для остальных bias, единицы для весов LayerNorm.

Больше ничего в пакет не входит: `TransformerEncoder`, оптимизаторы,
autograd-движок и прочие слои удалены.

## Структура

```
part_2_transformer/
├── main.py                  # parity-демо и синхронное обучение с torch
├── main.ipynb               # то же в ноутбуке + график обучения
├── README.md
├── src/nptorch/
│   ├── __init__.py          # TransformerEncoderLayer, init, nn
│   ├── init.py              # xavier/kaiming/uniform/constant
│   └── nn/
│       ├── __init__.py
│       ├── ops.py           # linear, layer_norm, dropout, softmax, relu, gelu
│       └── transformer.py   # MHA, FFN, TransformerEncoderLayer
└── tests/
    ├── test_ops.py          # операции против torch
    ├── test_layer.py        # forward/backward parity, маски, API
    ├── test_training.py     # синхронное обучение с torch
    └── utils.py
```

## Быстрый старт

```python
import numpy as np
from src.nptorch import TransformerEncoderLayer

layer = TransformerEncoderLayer(
    d_model=8, nhead=2, dim_feedforward=16, dropout=0.0
).eval()

x = np.random.default_rng(0).normal(size=(4, 3, 8))  # (L, N, E)
out = layer(x)

# Отдельный обратный проход: forward уже сохранён в кэше.
layer.zero_grad()
dout = np.random.default_rng(1).normal(size=out.shape)
dx = layer.backward(dout)

print(out.shape, dx.shape)
print(sorted(layer.grads))       # градиенты всех параметров
for name, parameter in layer.named_parameters():
    parameter -= 0.01 * layer.grads[name]   # обычное SGD-обновление
```

`forward` и `backward` — два отдельных вызова. `backward` использует кэш
последнего `forward`; повторный `forward` перезаписывает кэш. Повторный
`backward` без нового `forward` **накапливает** градиенты, как torch.

## Математика

### Forward

Для входа $X \in \mathbb{R}^{L \times N \times E}$ и $d_h = E/H$:

$$
Q = XW_q^\top+b_q,\quad K = XW_k^\top+b_k,\quad V = XW_v^\top+b_v
$$

$$
S_h = \frac{Q_hK_h^\top}{\sqrt{d_h}} + M,\qquad A_h = \operatorname{softmax}(S_h)
$$

$$
A_h^d = \operatorname{dropout}(A_h),\qquad
\operatorname{MHA}(X) = \operatorname{concat}_h(A_h^dV_h)W_o^\top+b_o,\qquad
\operatorname{SA}(X) = \operatorname{dropout}(\operatorname{MHA}(X))
$$

$$
\operatorname{FF}(X) = \operatorname{dropout}\left(
W_2\operatorname{dropout}\left(\sigma(W_1X+b_1)\right)+b_2\right)
$$

Post-norm:

$$
Z = \operatorname{LN}_1(X+\operatorname{SA}(X)),\qquad
Y = \operatorname{LN}_2(Z+\operatorname{FF}(Z))
$$

Pre-norm:

$$
Z = X+\operatorname{SA}(\operatorname{LN}_1(X)),\qquad
Y = Z+\operatorname{FF}(\operatorname{LN}_2(Z))
$$

Layer norm для строки $x\in\mathbb{R}^D$:

$$
\mu = \frac1D\sum_i x_i,\qquad
\sigma = \sqrt{\frac1D\sum_i(x_i-\mu)^2+\varepsilon},\qquad
y_i = \gamma_i\frac{x_i-\mu}{\sigma}+\beta_i
$$

### Backward

Chain rule на уровне слоя (post-norm):

$$
\bar U_2 = \operatorname{LN}_2'(\bar Y),\qquad
\bar Z = \bar U_2 + \operatorname{FF}'(\bar U_2),
$$

$$
\bar U_1 = \operatorname{LN}_1'(\bar Z),\qquad
\bar X = \bar U_1 + \operatorname{SA}'(\bar U_1).
$$

Pre-norm:

$$
\bar U_1 = \bar Y + \operatorname{LN}_2'(\operatorname{FF}'(\bar Y)),
\qquad
\bar X = \bar U_1 + \operatorname{LN}_1'(\operatorname{SA}'(\bar U_1)).
$$

Ключевые формулы, полностью выведенные в docstring-ах:

- линейный слой: $\bar X = \bar YW$, $\bar W = \bar Y^\top X$, $\bar b=\sum_r\bar Y_r$;
- layer norm:
  $\bar x_i = \frac{1}{\sigma}\left[g_i-\frac1D\sum_j g_j-\frac{\hat x_i}{D}\sum_j g_j\hat x_j\right]$,
  где $g_i=\bar y_i\gamma_i$;
- softmax: $\bar s_i = p_i\left(\bar p_i-\sum_j p_j\bar p_j\right)$;
- attention: $\bar A_h^d = \bar O_hV_h^\top$, $\bar V_h=(A_h^d)^\top\bar O_h$,
  $\bar A_h = \operatorname{dropout}'(\bar A_h^d)$,
  $\bar S_h = A_h\odot(\bar A_h-\sum_j A_{h,ij}\bar A_{h,ij})$,
  $\bar Q_h=\frac{1}{\sqrt{d_h}}\bar S_hK_h$,
  $\bar K_h=\frac{1}{\sqrt{d_h}}\bar S_h^\top Q_h$;
- dropout: $\bar x_i = m_i\bar y_i/(1-p)$;
- ReLU: $\bar x_i=\bar y_i\mathbf{1}[x_i>0]$;
- GELU: $\bar x_i=\bar y_i(\Phi(x_i)+x_i\varphi(x_i))$.

## Маски

- `src_mask`: `(L, S)` или `(N*H, L, S)`. Bool: `True` = запрещённая позиция.
  Float: аддитивная маска (например, `-inf`).
- `src_key_padding_mask`: `(N, S)` для батча, `(S,)` для unbatched входа
  (дополнительно принимается `(1, S)`, torch требует только `(S,)`).
  Bool: `True` = padding.
- `is_causal=True` требует переданную `src_mask` (как в torch MHA). Если
  `src_key_padding_mask` не задан, вместо содержимого `src_mask` применяется
  верхнетреугольная причинная маска; если padding задан, используется
  `src_mask`, а флаг игнорируется. Это именно поведение torch 2.15.
- Полностью замаскированная строка запроса получает нулевые веса внимания и
  нулевой градиент по score-строке вместо `NaN`. Градиент по входу такой
  позиции при этом не обязан быть нулём: её value-вектор используют другие
  запросы.

## Проверка с torch

`main.py` сравнивает NumPy-слой и `torch.nn.TransformerEncoderLayer` с
одинаковыми весами (`float64`, CPU, `dropout=0`). Сравниваются выход,
$\partial L/\partial X$ и градиенты **всех** параметров:

```
configuration     forward      d_src   d_params
post-norm        6.66e-16   1.33e-15   1.78e-15
pre-norm         4.44e-16   3.61e-16   8.88e-16
gelu             4.44e-16   8.88e-16   2.22e-15
no bias          8.88e-16   1.33e-15   1.78e-15
masks            6.66e-16   1.33e-15   1.78e-15
causal           4.44e-16   8.88e-16   2.00e-15
```

`tests/test_layer.py` независимо от torch проверяет **все** компоненты
градиента конечными разностями (центральная разность, `eps=1e-6`), включая
pre-norm, post-norm и dropout с зафиксированными масками. Для dropout=0
перебираются все параметры и все элементы `dX`; для dropout>0 — также все
параметры и все элементы `dX` при воспроизводимых масках.

## Демо обучения

`main.py` обучает один слой регрессии на `tanh(XR+b)` и синхронно обновляет
torch-близнеца теми же градиентами и тем же SGD:

```
loss: 1.406524 -> 0.287498 (20.4% of the initial value)
max |gradient diff| during training: 1.67e-16
max |parameter diff| after updates:  2.22e-16
state_dict keys match torch: True
```

Ноутбук [`main.ipynb`](main.ipynb) содержит те же сравнения, проверку
конечными разностями и график loss для обеих реализаций.

## Запуск

```bash
cd part_2_transformer

# тесты
../.venv/bin/python -m unittest discover -s tests -t . -v

# покрытие
COVERAGE_FILE=/tmp/part2.coverage ../.venv/bin/python -m coverage run \
    --branch --source=src/nptorch -m unittest discover -s tests -t .
COVERAGE_FILE=/tmp/part2.coverage ../.venv/bin/python -m coverage report -m

# демо
../.venv/bin/python main.py

# ноутбук
../.venv/bin/jupyter notebook main.ipynb
```

Текущий результат: **80 тестов**, `100%` покрытия операторов и ветвлений
(`458` statements, `142` branches).

## Отличия от torch и ограничения

- Все вычисления в `float64`; параметра `dtype`/`device` нет. Вход `float32`
  приводится к `float64`.
- Поддерживаются только `activation="relu"` и `activation="gelu"`; callable
  принимается, если его имя `relu`/`gelu`.
- `dropout` использует глобальный `numpy.random`, а не RNG torch. Точное
  сравнение с torch выполняется при `dropout=0` или в режиме `eval()`.
- Нет nested tensors, `add_bias_kv`, `add_zero_attn`, отдельных `kdim`/`vdim`
  и `TransformerEncoder` — они не нужны одному слою.
- Полностью замаскированные строки дают нулевые веса внимания и нулевой
  score-градиент (safe SDPA convention). Стандартный путь torch для таких
  строк возвращает такие же конечные значения; `NaN` возможен только в
  нативном fastpath инференса torch (`eval`, `batch_first=True`, 3-D вход,
  параметры без `requires_grad`, не более одной маски).
- `is_causal` — подсказка, а не независимая причинная маска; см. раздел
  «Маски». NumPy проверяет форму `src_mask` даже тогда, когда hint-режим
  без padding её игнорирует; torch в этом нестандартном случае может
  проигнорировать некорректную форму. Целочисленные маски дают `TypeError`
  (в torch — `AssertionError`).
- После `forward` нельзя менять `params` или вход in-place до `backward`:
  кэш хранит ссылки, в отличие от version counter в torch.
