# Copyright (c) 2026 azim-givron
#
# This source code is licensed under the MIT license.
# SPDX-License-Identifier: MIT
import unittest

import torch
from flow_matching.path import DirichletSmoothing


class TestDirichletSmoothing(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.num_classes = 4
        self.classes = torch.randint(0, self.num_classes, (32, 6))
        self.labels = torch.nn.functional.one_hot(self.classes, self.num_classes).to(
            torch.float64
        )
        self.smoothing = DirichletSmoothing()

    def test_bounds(self):
        x = self.smoothing.smooth(self.labels)
        nu = self.smoothing.interpolation_weight
        tau = self.smoothing.boundary_mass / self.num_classes
        self.assertTrue(torch.allclose(x.sum(dim=-1), torch.ones(32, 6, dtype=x.dtype)))
        self.assertTrue(bool((x > (1 - nu) * tau).all()))
        label_coordinate = torch.gather(x, -1, self.classes.unsqueeze(-1))
        self.assertTrue(bool((label_coordinate >= nu).all()))
        self.assertTrue(torch.equal(x.argmax(dim=-1), self.classes))

    def test_deterministic(self):
        first = self.smoothing.smooth(self.labels, stochastic=False)
        second = self.smoothing(self.labels, stochastic=False)
        self.assertTrue(torch.equal(first, second))
        nu = self.smoothing.interpolation_weight
        expected = nu * self.labels + (1 - nu) / self.num_classes
        self.assertTrue(torch.allclose(first, expected))

    def test_log_prob_is_normalized(self):
        # For K = 2 the simplex is a segment parametrized by its first coordinate.
        smoothing = DirichletSmoothing(dirichlet_alpha=5.0)
        label = torch.tensor([1.0, 0.0], dtype=torch.float64)
        nu, mass = smoothing.interpolation_weight, smoothing.boundary_mass
        low = nu + (1 - nu) * mass / 2
        high = low + (1 - nu) * (1 - mass)
        s = torch.linspace(low, high, 20001, dtype=torch.float64)[1:-1]
        x = torch.stack([s, 1 - s], dim=-1)
        density = smoothing.log_prob(x, label.expand_as(x)).exp()
        self.assertAlmostEqual(torch.trapezoid(density, s).item(), 1.0, places=3)

    def test_log_prob_outside_support(self):
        x = torch.full((1, self.num_classes), 1.0 / self.num_classes)
        with self.assertRaises(ValueError):
            self.smoothing.log_prob(x, self.labels[:1, 0].float())

    def test_validation(self):
        with self.assertRaises(ValueError):
            DirichletSmoothing(interpolation_weight=0.4)
        with self.assertRaises(ValueError):
            DirichletSmoothing(dirichlet_alpha=0.0)
        with self.assertRaises(ValueError):
            DirichletSmoothing(boundary_mass=1.0)


if __name__ == "__main__":
    unittest.main()
