# Lab 2. TransformerEncoderLayer и TransformerEncoder на NumPy

Реализация `torch.nn.TransformerEncoderLayer` с дефолтными параметрами
(`d_model=512`, `nhead=8`, `dim_feedforward=2048`, `dropout=0.1`,
`activation=relu`, `layer_norm_eps=1e-5`, `batch_first=False`,
`norm_first=False`, `bias=True`) и стека `torch.nn.TransformerEncoder`
на чистом NumPy: forward и backward написаны вручную, без `torch.autograd`.

Чтобы не писать разрозненные `backward` для каждого слоя, сначала сделан
небольшой torch-подобный autograd-стек:

- `Tensor` с `requires_grad`, `grad`, `backward()`, `detach()`, `no_grad()`;
- функциональные операции с аналитическими градиентами: арифметика,
  broadcasting, `matmul`, `softmax`, `layer_norm`, `gelu`, `dropout`,
  маски, редукции, reshape/transpose/permute/slice;
- `nn.Module`, `nn.ModuleList`, `Parameter`, `state_dict` / `load_state_dict`;
- `nn.Linear`, `nn.LayerNorm`, `nn.Dropout`, `nn.ReLU`, `nn.GELU`,
  `nn.MultiheadAttention`, `nn.MSELoss`;
- `nn.TransformerEncoderLayer` и `nn.TransformerEncoder` (клонирование
  слоёв, финальный `norm`, канонизация масок, автоопределение `is_causal`);
- `optim.SGD`, `optim.Adam`, `optim.AdamW`.

Интерфейсы и имена параметров специально совместимы с torch: веса можно
загрузить напрямую из `torch.nn.TransformerEncoderLayer.state_dict()`.

## Структура

```
part_2_transformer/
├── src/nptorch/
│   ├── tensor.py          # Tensor, autograd-движок, no_grad
│   ├── functional.py      # операции и ручные backward-формулы
│   ├── init.py            # xavier/kaiming/uniform как в torch.nn.init
│   ├── optim.py           # SGD, Adam, AdamW
│   └── nn/
│       ├── module.py      # Module, state_dict, train/eval
│       ├── container.py   # ModuleList
│       ├── linear.py
│       ├── normalization.py
│       ├── dropout.py
│       ├── activation.py
│       ├── attention.py   # MultiheadAttention
│       ├── transformer.py # TransformerEncoderLayer + TransformerEncoder
│       └── loss.py
├── tests/
│   ├── test_autograd.py   # операции против torch и конечных разностей
│   ├── test_nn.py         # слои и MHA
│   ├── test_transformer.py # главный тест: слой против torch
│   ├── test_encoder.py     # стек слоёв против torch
│   ├── test_modulelist.py  # контейнер подмодулей
│   ├── test_api.py        # публичный API и error paths
│   └── test_optim.py      # оптимизаторы против torch.optim
├── main.py                # демонстрация паритета и обучения
└── main.ipynb             # исполненный ноутбук: паритет + обучение XOR
```

## Запуск

Из директории `part_2_transformer` (venv лежит в корне проекта):

```bash
../.venv/bin/python -m unittest discover -s tests -t . -v
```

Сейчас в наборе 112 тестов; полный прогон занимает около 1.5 секунды.

### Покрытие

`coverage` поставлен в venv как dev-инструмент (в `pyproject.toml` не добавлялся):

```bash
COVERAGE_FILE=/tmp/part2.coverage ../.venv/bin/python -m coverage run \
    --branch --source=src/nptorch -m unittest discover -s tests -t .
COVERAGE_FILE=/tmp/part2.coverage ../.venv/bin/python -m coverage report -m
```

На момент проверки: **97.2% строк, 88.2% веток, 94.8% combined**. Непокрытыми
остаются в основном защитные проверки аргументов и редкие ошибки; ключевая
математика (autograd, attention, слои, encoder) покрыта паритетными тестами
против torch.

Быстрая демонстрация:

```bash
../.venv/bin/python main.py
```

Ноутбук с уже сохранёнными выводами:

```bash
../.venv/bin/jupyter notebook main.ipynb
```

В `main.ipynb` три части: паритет `TransformerEncoderLayer`, паритет
`TransformerEncoder` и обучение tiny-классификатора на XOR первых/последних
токенов (300 шагов, ~3.5 секунды на CPU, график loss/accuracy). В конце веса
обученной NumPy-модели загружаются в torch-двойник и логиты сравниваются
(расхождение ~1e-15).

Отдельный тест:

```bash
../.venv/bin/python -m unittest tests.test_transformer -v
```

## Что проверяется

Тесты сравнивают не только выход, но и **все градиенты**: по входу и по
каждому параметру. Для этого в torch-модель загружаются те же веса, что и
в numpy-модель, а loss задаётся одинаковым случайным коэффициентом.

| Проверка | Режим | Максимальное расхождение |
|---|---|---|
| Дефолтный слой `512/8/2048` | float32, eval, forward | ~1e-6 |
| Дефолтный слой, loss | float32, eval | ~1e-5 (отн. ~1e-7) |
| Градиент входа | float32, eval | ~1e-6 |
| Градиенты всех параметров | float32, eval | ~5e-6 |
| Маленький слой | float64, eval, forward/backward | ~1e-15 |
| Маски `src_mask` / `src_key_padding_mask` | float64, bool и float | ~1e-15 |
| `batch_first`, `norm_first`, `gelu`, `bias=False` | float64 | ~1e-15 |
| Все параметры слоя | конечные разности, float64 | < 2e-6 (порог теста) |
| 3 шага SGD, параметры после шага | float64 | ~1e-16 |
| `MultiheadAttention`, включая `need_weights` | float64 | ~1e-15 |
| `TransformerEncoder` (3 слоя + norm, маски) | float64 | ~1e-15 |
| `TransformerEncoder` (2 слоя) | float32 | ~1e-6 |

Конечные разности считаются полностью независимо от torch: аналитический
градиент сравнивается с центральной разностью по каждому элементу входа и
каждому параметру. Это ловит ошибки, которые не заметны при сравнении двух
реализаций с одинаковой неверной формулой.

Дропаут в train-режиме нельзя сравнить с torch по значениям: RNG у NumPy и
torch разный. Поэтому:

- весь слой сравнивается в `eval()` (dropout выключен) на дефолтном
  `dropout=0.1`;
- отдельно проверяется train-режим с `dropout=0.0` (точный паритет);
- для `Dropout` есть тест с явно заданной маской, где проверяется
  inverted-dropout формула и градиент;
- train-режим с `dropout>0` проверяется на конечность и форму градиентов.

## Замечания по реализации

- **Autograd.** Каждая операция сохраняет замыкание, которое добавляет
  градиенты родителям. `backward()` делает топологическую сортировку.
  Промежуточные градиенты сбрасываются на каждом проходе, а листовые
  тензоры накапливают градиент, как в torch.
- **Broadcasting.** Для `add/mul/...` градиент суммируется обратно к форме
  входа (`_unbroadcast`), поэтому bias и маски работают без ручных циклов.
- **dtype.** Python-скаляры не поднимают `float32` до `float64` (правила
  NEP 50 и поведение torch) и конвертируются сразу в dtype второго
  операнда, поэтому `float64` не теряет точность из-за промежуточного
  `float32`. Type promotion для int/bool/float следует правилам torch;
  есть регрессионные тесты на `head_dim=8` (непредставимый в float32
  масштаб внимания) и на int-тензоры.
- **Полностью замаскированные строки.** При `need_weights=False` (путь
  `TransformerEncoderLayer`) внимание для такой строки равно нулю, как в
  SDPA; при `need_weights=True` сохраняется NaN, как в медленном пути
  torch.
- **Производительность.** `F.linear` сплющивает ведущие размерности в один
  2-D GEMM: `(L, N, E) @ (E, O)` одним вызовом BLAS заметно быстрее, чем
  `L` маленьких batched-matmul. Это дало ускорение forward примерно в 5
  раз и backward примерно в 7 раз на дефолтном слое.
- **Multi-head attention** реализован медленным ручным путём: проекции,
  reshape по головам, маски, softmax, dropout, `out_proj`. Это совпадает с
  `torch.nn.functional.multi_head_attention_forward` и даёт полный
  backward-граф. `add_bias_kv`, `add_zero_attn` и разные `kdim/vdim`
  намеренно не реализованы (для них бросается `NotImplementedError`).
- **`is_causal`.** Как и в torch 2.15, `is_causal=True` требует переданного
  `attn_mask`-намёка; без него бросается `RuntimeError`. У `TransformerEncoder`
  при `is_causal=None` маска сравнивается со стандартной каузальной и флаг
  выставляется автоматически, как в torch.
- **`TransformerEncoder`.** Слои клонируются через `copy.deepcopy`, поэтому
  стартуют с одинаковыми весами (как в torch), но не разделяют параметры.
  Поддерживаются финальный `norm`, bool/float маски, 2D/3D вход,
  `batch_first`, `norm_first`, `gelu`, `bias=False`. Nested tensor в NumPy нет:
  `enable_nested_tensor` и `mask_check` принимаются для совместимости API, но
  `use_nested_tensor` всегда `False`.
- **Numba.** В горячем пути не используется. Локальные замеры row-wise
  ядер дали ~1.1x для softmax и ~1.7x для layer_norm на форме `(256, 512)`,
  но на фоне BLAS-матмулов это несколько процентов времени слоя, а numba
  добавляет компиляцию и второй набор backward-формул. Поэтому оставлен
  чистый NumPy; отдельного numba-бэкенда нет.

## Ограничения

- Только CPU, float32/float64; CUDA не используется и не устанавливается.
- Нет `retain_graph`/`create_graph`, второго порядка и in-place операций.
- Нет fused SDPA/flash-attention; используется ручная математика внимания.
- `MultiheadAttention` не поддерживает `add_bias_kv`, `add_zero_attn`,
  `kdim != embed_dim`, `vdim != embed_dim`.
- `ModuleList.insert` следует семантике Python-list для отрицательных и
  выходящих за границу индексов; torch 2.15 в этих edge-случаях ведёт себя
  иначе (KeyError/дырки в ключах). Валидные `insert(0..len)` совпадают.
- У оптимизаторов нет `state_dict()/load_state_dict()/add_param_group()`;
  поддерживается один `param_group`.
- Nested tensor fast path `TransformerEncoder` не реализован (NumPy не умеет
  ragged tensors). Для точного сравнения передавай в torch
  `enable_nested_tensor=False`; иначе torch может обнулять выходы в padding-
  позициях, а numpy-версия считает их обычным образом.
- Dropout использует собственный NumPy RNG; для воспроизводимости есть
  `nptorch.manual_seed(seed)`.
