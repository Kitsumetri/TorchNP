# Математические выкладки для `TransformerEncoderLayer`

Этот файл содержит полные формулы forward/backward, которые реализованы в
[`transformer.py`](src/nptorch/nn/transformer.py) и
[`ops.py`](src/nptorch/nn/ops.py). Автограда в проекте нет: каждый backward
получен явным дифференцированием, а не автоматическим построением графа.

## 0. Обозначения

- `L` — длина последовательности, `N` — размер батча, `E` — размер модели.
- `H` — число голов, `d_h = E / H`.
- `F` — размер скрытого слоя FFN (`dim_feedforward`).
- Внутренняя раскладка — sequence-first: `X ∈ R^{L×N×E}`. При
  `batch_first=True` вход и выход транспонируются, на математику это не влияет.
- `⊙` — поэлементное произведение; `Σ_r` — сумма по всем ведущим индексам
  (батч и последовательность), если не указано иное.
- `𝟏` — вектор из единиц; `softmax` применяется к последней оси.
- `M` — аддитивная маска: `0` на разрешённых позициях, `-∞` на запрещённых.
  Bool-маска канонизируется в `M`; `src_mask`, `src_key_padding_mask` и
  причинная маска складываются.

Все формулы ниже записаны для одного слоя с bias. При `bias=False`
соответствующие векторы `b` и их градиенты просто отсутствуют.

## 1. Линейное преобразование

Прямой проход:

$$
Y = XW^\top + b,
\qquad
y_o = \sum_i W_{oi}x_i + b_o.
$$

Здесь `X` имеет форму `(..., D_in)`, `W` — `(D_out, D_in)`, `b` — `(D_out,)`.

Пусть `Ȳ = ∂L/∂Y`. Дифференцируя по элементам:

$$
\frac{\partial L}{\partial x_i}
= \sum_o \bar y_o W_{oi}
\quad\Rightarrow\quad
\bar X = \bar YW,
$$

$$
\frac{\partial L}{\partial W_{oi}}
= \sum_r \bar y_{r,o}x_{r,i}
\quad\Rightarrow\quad
\bar W = \bar Y_{\mathrm{flat}}^\top X_{\mathrm{flat}},
$$

$$
\frac{\partial L}{\partial b_o}
= \sum_r \bar y_{r,o}
\quad\Rightarrow\quad
\bar b = \sum_r \bar Y_r.
$$

`X_flat` и `Y_flat` — массивы, в которых все ведущие индексы схлопнуты в один.

## 2. Layer normalization

Для каждой строки `x ∈ R^D` (нормализуется последняя ось):

$$
\mu = \frac{1}{D}\sum_{i=1}^D x_i,
\qquad
v = \frac{1}{D}\sum_{i=1}^D (x_i-\mu)^2,
\qquad
\sigma = \sqrt{v+\varepsilon},
$$

$$
\hat x_i = \frac{x_i-\mu}{\sigma},
\qquad
y_i = \gamma_i\hat x_i+\beta_i.
$$

### 2.1. Производные forward-величин

Считая `μ` и `σ` функциями от `x`:

$$
\frac{\partial \mu}{\partial x_i} = \frac{1}{D},
$$

$$
\frac{\partial v}{\partial x_i}
= \frac{2}{D}(x_i-\mu)
\quad\Rightarrow\quad
\frac{\partial \sigma}{\partial x_i}
= \frac{x_i-\mu}{D\sigma}
= \frac{\hat x_i}{D}.
$$

Для `x̂_j = (x_j-μ)/σ`:

$$
\frac{\partial \hat x_j}{\partial x_i}
= \frac{\delta_{ij}}{\sigma}
- \frac{1}{D\sigma}
- \frac{\hat x_i\hat x_j}{D\sigma}.
$$

### 2.2. Backward

Обозначим `g_i = ȳ_i γ_i`. Тогда

$$
\bar x_i
= \sum_j g_j\frac{\partial \hat x_j}{\partial x_i}
= \frac{1}{\sigma}\left[
    g_i
    - \frac{1}{D}\sum_j g_j
    - \frac{\hat x_i}{D}\sum_j g_j\hat x_j
  \right].
$$

В матричной форме:

$$
\bar X
= \frac{\mathrm{rstd}}{D}\left(
    D\,G - \sum_j G_j
    - \hat X\odot\sum_j (G\odot\hat X)_j
  \right),
\qquad
G = \bar Y\odot\gamma,
$$

где суммы берутся по последней оси с сохранением размерности.

Градиенты параметров:

$$
\bar\gamma_i = \sum_r \bar y_{r,i}\hat x_{r,i},
\qquad
\bar\beta_i = \sum_r \bar y_{r,i}.
$$

Если `bias=False`, `β` отсутствует, а `β̄` отбрасывается.

## 3. Softmax

По последней оси:

$$
p_i = \frac{e^{s_i}}{\sum_k e^{s_k}}
= \frac{e^{s_i-m}}{\sum_k e^{s_k-m}},
\qquad
m = \max_k s_k.
$$

Сдвиг на максимум не меняет результат и нужен для численной устойчивости.

Дифференциал:

$$
\frac{\partial p_j}{\partial s_i}
= p_j(\delta_{ji}-p_i).
$$

Поэтому для `p̄ = ∂L/∂p`:

$$
\bar s_i
= \sum_j \bar p_j\frac{\partial p_j}{\partial s_i}
= p_i\left(\bar p_i-\sum_j p_j\bar p_j\right).
$$

Если позиция замаскирована, `p_j = 0`, значит `s̄_j = 0`; градиент в
замаскированный score не течёт.

### 3.1. Полностью замаскированная строка

Если все ключи строки замаскированы, по соглашению проекта

$$
p_i = 0 \quad \forall i,
\qquad
\bar s_i = 0 \quad \forall i.
$$

При этом `V̄ = P^\top Ō` не получает вклада от такой строки, но `X̄` для
соответствующей позиции может быть ненулевым: её value-вектор используется
другими запросами.

## 4. Dropout

Inverted dropout. Для независимых `m_i ∼ Bernoulli(1-p)`:

$$
y_i = \frac{m_i x_i}{1-p},
\qquad
\mathbb{E}[y_i] = x_i.
$$

Backward:

$$
\bar x_i = \frac{m_i}{1-p}\bar y_i.
$$

При `p=1` forward возвращает нули, а `x̄ = 0`. В режиме `eval()` dropout
является тождественным отображением: `m_i = 1`, `x̄ = ȳ`.

## 5. Активации

### 5.1. ReLU

$$
f(x) = \max(0,x),
\qquad
f'(x) = \mathbf{1}[x>0],
\qquad
\bar x_i = \bar y_i\mathbf{1}[x_i>0].
$$

При `x_i=0` производная доопределена нулём, как в PyTorch.

### 5.2. Точный GELU

Пусть

$$
\Phi(x) = \frac12\left(1+\mathrm{erf}\left(\frac{x}{\sqrt2}\right)\right),
\qquad
\varphi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}.
$$

Тогда `Φ'(x) = φ(x)` и

$$
f(x) = x\Phi(x),
$$

$$
f'(x) = \Phi(x) + x\varphi(x),
\qquad
\bar x_i = \bar y_i\left(\Phi(x_i)+x_i\varphi(x_i)\right).
$$

## 6. Multi-head self-attention

### 6.1. Forward

Вход attention-блока — `X ∈ R^{L×N×E}`. Параметры in-projection хранятся
одной матрицей

$$
W_{in} = \begin{bmatrix}W_q\\W_k\\W_v\end{bmatrix}
\in R^{3E×E},
\qquad
b_{in} = \begin{bmatrix}b_q\\b_k\\b_v\end{bmatrix}
\in R^{3E},
$$

где `W_q, W_k, W_v ∈ R^{E×E}`.

Проекции:

$$
Q = XW_q^\top+b_q,
\qquad
K = XW_k^\top+b_k,
\qquad
V = XW_v^\top+b_v.
$$

Разбиение на головы: `Q_h, K_h, V_h ∈ R^{(N·H)×L×d_h}` — результат
reshape `(L,N,H,d_h)` и транспонирования в `(N·H,L,d_h)`.

Score и веса:

$$
S_h = \frac{Q_hK_h^\top}{\sqrt{d_h}} + M_h,
\qquad
A_h = \mathrm{softmax}(S_h),
$$

где `M_h` — сумма broadcast-масок: `src_mask`, `src_key_padding_mask`,
причинная маска (если она применяется). Bool-маски предварительно заменяются
на `0/-∞`.

Dropout весов внимания:

$$
A_h^d = \mathrm{dropout}(A_h).
$$

Контекст и выход голов:

$$
O_h = A_h^d V_h
\in R^{(N·H)×L×d_h},
$$

$$
O = \mathrm{merge}(O_1,\ldots,O_H)
\in R^{L×N×E},
$$

$$
U = OW_o^\top+b_o,
\qquad
\mathrm{SA}(X) = \mathrm{dropout}(U).
$$

### 6.2. Backward

Пусть `Ȳ = ∂L/∂SA(X)`.

**Выходной dropout:**

$$
\bar U = \mathrm{dropout}'(\bar Y).
$$

**Выходная проекция:**

$$
\bar O = \bar UW_o,
\qquad
\bar W_o = \bar U_{\mathrm{flat}}^\top O_{\mathrm{flat}},
\qquad
\bar b_o = \sum_r \bar U_r.
$$

**Головы.** Разобьём `Ō` обратно на `Ō_h`:

$$
\bar A_h^d = \bar O_hV_h^\top,
\qquad
\bar V_h = (A_h^d)^\top\bar O_h.
$$

**Dropout весов внимания:**

$$
\bar A_h = \mathrm{dropout}'(\bar A_h^d).
$$

**Softmax:**

$$
\bar S_h = A_h\odot\left(
  \bar A_h
  - \left(\sum_j A_{h,ij}\bar A_{h,ij}\right)\mathbf 1^\top
\right).
$$

Сумма в скобках берётся по ключевой оси `j` для каждой строки `i`.

**Score и Q/K:**

$$
\bar Q_h = \frac{1}{\sqrt{d_h}}\bar S_hK_h,
\qquad
\bar K_h = \frac{1}{\sqrt{d_h}}\bar S_h^\top Q_h.
$$

После merge-обратного преобразования получаем `Q̄, K̄, V̄` в форме
`(L,N,E)`.

**In-projection.** Так как `Q`, `K`, `V` считаются от одного `X`:

$$
\bar X = \bar QW_q + \bar KW_k + \bar VW_v,
$$

$$
\bar W_q = \bar Q_{\mathrm{flat}}^\top X_{\mathrm{flat}},
\qquad
\bar W_k = \bar K_{\mathrm{flat}}^\top X_{\mathrm{flat}},
\qquad
\bar W_v = \bar V_{\mathrm{flat}}^\top X_{\mathrm{flat}},
$$

$$
\bar b_q = \sum_r \bar Q_r,
\qquad
\bar b_k = \sum_r \bar K_r,
\qquad
\bar b_v = \sum_r \bar V_r.
$$

Матрицы складываются в том же порядке, что и в forward:

$$
\bar W_{in} =
\begin{bmatrix}\bar W_q\\ \bar W_k\\ \bar W_v\end{bmatrix},
\qquad
\bar b_{in} =
\begin{bmatrix}\bar b_q\\ \bar b_k\\ \bar b_v\end{bmatrix}.
$$

Маски — константы, поэтому градиентов по ним нет. Для замаскированных
позиций `A_{h,ij}=0`, и формула softmax-backward автоматически даёт
`S̄_{h,ij}=0`.

## 7. Position-wise feed-forward network

### 7.1. Forward

$$
H = XW_1^\top+b_1,
\qquad
A = \sigma(H),
$$

$$
D_1 = \mathrm{dropout}(A),
\qquad
P = D_1W_2^\top+b_2,
$$

$$
\mathrm{FF}(X) = \mathrm{dropout}(P).
$$

Здесь `W_1 ∈ R^{F×E}`, `W_2 ∈ R^{E×F}`, `σ` — ReLU или точный GELU.

### 7.2. Backward

Пусть `Ȳ` — градиент по выходу FFN.

$$
\bar P = \mathrm{dropout}'(\bar Y),
$$

$$
\bar D_1 = \bar PW_2,
\qquad
\bar W_2 = \bar P_{\mathrm{flat}}^\top (D_1)_{\mathrm{flat}},
\qquad
\bar b_2 = \sum_r \bar P_r,
$$

$$
\bar A = \mathrm{dropout}'(\bar D_1),
$$

$$
\bar H = \bar A\odot\sigma'(H),
$$

$$
\bar X = \bar HW_1,
\qquad
\bar W_1 = \bar H_{\mathrm{flat}}^\top X_{\mathrm{flat}},
\qquad
\bar b_1 = \sum_r \bar H_r.
$$

## 8. Сборка слоя

Обозначим через `LN'(G)` результат backward layer norm при входном градиенте
`G`; через `SA'(G)` и `FF'(G)` — аналогичные backward-функции блоков
внимания и FFN. Каждая из них также накапливает градиенты своих параметров.

### 8.1. Post-norm (по умолчанию)

Forward:

$$
U_1 = X+\mathrm{SA}(X),
\qquad
Z = \mathrm{LN}_1(U_1),
$$

$$
U_2 = Z+\mathrm{FF}(Z),
\qquad
Y = \mathrm{LN}_2(U_2).
$$

Backward при `Ȳ = ∂L/∂Y`:

$$
\bar U_2 = \mathrm{LN}_2'(\bar Y),
$$

$$
\bar Z = \bar U_2 + \mathrm{FF}'(\bar U_2),
$$

$$
\bar U_1 = \mathrm{LN}_1'(\bar Z),
$$

$$
\bar X = \bar U_1 + \mathrm{SA}'(\bar U_1).
$$

Пояснение: `FF'` возвращает градиент по своему входу `Z`, а остаточная
связь `U_2 = Z + FF(Z)` добавляет ещё один путь `Ū_2`; поэтому
`Z̄ = Ū_2 + FF'(Ū_2)`. Аналогично `X̄ = Ū_1 + SA'(Ū_1)`.

### 8.2. Pre-norm (`norm_first=True`)

Forward:

$$
A_1 = \mathrm{LN}_1(X),
\qquad
U_1 = X+\mathrm{SA}(A_1),
$$

$$
A_2 = \mathrm{LN}_2(U_1),
\qquad
Y = U_1+\mathrm{FF}(A_2).
$$

Backward:

$$
\bar U_1
= \bar Y + \mathrm{LN}_2'\!\left(\mathrm{FF}'(\bar Y)\right),
$$

$$
\bar X
= \bar U_1
+ \mathrm{LN}_1'\!\left(\mathrm{SA}'(\bar U_1)\right).
$$

Здесь `FF'(Ȳ)` — градиент по `A_2`, затем `LN_2'` переводит его в градиент
по `U_1`. Аналогично `SA'(Ū_1)` даёт градиент по `A_1`, а `LN_1'` — по `X`.

## 9. Инициализация

Для весовой матрицы формы `(D_out, D_in)`:

$$
\mathrm{fan}_{in} = D_{in},
\qquad
\mathrm{fan}_{out} = D_{out}.
$$

**Xavier/Glorot uniform** (используется для `in_proj_weight`):

$$
\mathrm{bound} = \sqrt{\frac{6}{\mathrm{fan}_{in}+\mathrm{fan}_{out}}},
\qquad
W_{ij}\sim U(-\mathrm{bound},\mathrm{bound}).
$$

**Kaiming uniform** с `a = √5` (используется для `out_proj` и линейных слоёв):

$$
\mathrm{bound} = \sqrt{\frac{6}{(1+a^2)\mathrm{fan}_{in}}}
= \frac{1}{\sqrt{\mathrm{fan}_{in}}},
\qquad
W_{ij}\sim U(-\mathrm{bound},\mathrm{bound}).
$$

**Bias:**

- `in_proj_bias`, `out_proj.bias`, `norm1.bias`, `norm2.bias` — нули;
- `linear1.bias`, `linear2.bias` — как в `torch.nn.Linear`:

$$
\mathrm{bound} = \frac{1}{\sqrt{\mathrm{fan}_{in}}},
\qquad
b_i\sim U(-\mathrm{bound},\mathrm{bound}).
$$

**LayerNorm:** `γ = 1`, `β = 0` (при наличии bias).

## 10. Маски и полностью замаскированные строки

- `src_mask`: форма `(L,S)` или `(N·H,L,S)`. Bool `True` = запрещено;
  float-значение прибавляется к score.
- `src_key_padding_mask`: `(N,S)` для батча, `(S,)` для unbatched входа.
  Bool `True` = padding.
- `is_causal=True` требует `src_mask` (как в torch MHA). Без
  `src_key_padding_mask` вместо `src_mask` применяется верхнетреугольная
  причинная маска; при наличии padding используется `src_mask`, а causal
  hint игнорируется.

Для строки запроса, у которой все ключи замаскированы, проект принимает
безопасное соглашение `A=0`, `S̄=0`. Это даёт конечные значения вместо
`NaN`; при этом `X̄` такой позиции не обязан быть нулём (см. §3.1).

## 11. Чем это проверено

- Полный набор тестов: 80 тестов, 100% statement/branch coverage.
- Forward/backward против `torch.nn.TransformerEncoderLayer` (float64, CPU):
  выход, `∂L/∂X` и градиенты всех параметров совпадают с точностью `~1e-15`.
- Независимые конечные разности без torch: все параметры и все элементы
  `∂L/∂X` при `dropout=0`, а также при `dropout>0` с воспроизводимыми
  масками.
- Покрыты post/pre-norm, `relu/gelu`, `bias=False`, `batch_first`,
  unbatched 2-D вход, 2-D/3-D bool/float маски, padding, causal hint.

Команда запуска тестов:

```bash
cd part_2_transformer
../.venv/bin/python -m unittest discover -s tests -t . -v
```
