"""Tests for the torch-like ModuleList container."""

from __future__ import annotations

import copy
import unittest

import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nptorch import nn


class TestModuleList(unittest.TestCase):
    def test_list_protocol(self):
        layers = nn.ModuleList(
            [nn.Linear(2, 2, dtype=np.float64) for _ in range(3)]
        )
        self.assertEqual(len(layers), 3)
        self.assertEqual(len(list(layers)), 3)
        self.assertIs(layers[0], layers[-3])
        self.assertIs(layers[-1], layers[2])
        with self.assertRaises(IndexError):
            _ = layers[3]
        subset = layers[1:]
        self.assertIsInstance(subset, nn.ModuleList)
        self.assertEqual(len(subset), 2)
        self.assertIs(subset[0], layers[1])

        replacement = nn.Linear(2, 2, dtype=np.float64)
        layers[1] = replacement
        self.assertIs(layers[1], replacement)
        with self.assertRaises(TypeError):
            layers[0] = object()
        with self.assertRaises(TypeError):
            layers[0] = np.zeros((2, 2))
        self.assertEqual(len(layers), 3)
        self.assertIn("0", layers._modules)
        layers.insert(1, nn.Linear(2, 2, dtype=np.float64))
        self.assertEqual(len(layers), 4)
        layers.insert(-1, nn.Linear(2, 2, dtype=np.float64))
        self.assertEqual(len(layers), 5)

        popped = layers.pop(-1)
        self.assertIsInstance(popped, nn.Linear)
        self.assertEqual(len(layers), 4)
        del layers[0]
        self.assertEqual(len(layers), 3)
        del layers[1:]
        self.assertEqual(len(layers), 1)

    def test_append_extend_add(self):
        layers = nn.ModuleList()
        layers.append(nn.Linear(2, 2, dtype=np.float64))
        layers.extend([nn.Linear(2, 2, dtype=np.float64) for _ in range(2)])
        self.assertEqual(len(layers), 3)
        layers += [nn.Linear(2, 2, dtype=np.float64)]
        self.assertEqual(len(layers), 4)
        combined = layers + [nn.Linear(2, 2, dtype=np.float64)]
        self.assertIsInstance(combined, nn.ModuleList)
        self.assertEqual(len(combined), 5)
        self.assertEqual(len(layers), 4)
        with self.assertRaises(TypeError):
            layers.append(object())
        with self.assertRaises(TypeError):
            layers.extend(3)

    def test_state_dict_keys_and_prefix(self):
        class Holder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Linear(2, 3, dtype=np.float64) for _ in range(2)]
                )

        holder = Holder()
        keys = list(holder.state_dict().keys())
        self.assertEqual(
            keys,
            [
                "layers.0.weight",
                "layers.0.bias",
                "layers.1.weight",
                "layers.1.bias",
            ],
        )
        self.assertEqual(
            [name for name, _ in holder.named_parameters()],
            ["layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias"],
        )

    def test_deepcopy(self):
        layers = nn.ModuleList([nn.Linear(2, 2, dtype=np.float64)])
        clone = copy.deepcopy(layers)
        self.assertIsNot(clone, layers)
        self.assertIsNot(clone[0], layers[0])
        self.assertIsNot(clone[0].weight, layers[0].weight)
        self.assertFalse(
            np.shares_memory(clone[0].weight.data, layers[0].weight.data)
        )
        self.assertTrue(
            np.array_equal(clone[0].weight.data, layers[0].weight.data)
        )


if __name__ == "__main__":
    unittest.main()
