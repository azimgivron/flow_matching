# Copyright (c) 2026 azim-givron
#
# This source code is licensed under the MIT license.
# SPDX-License-Identifier: MIT
import unittest

import torch
from flow_matching.utils import project_interior, project_simplex


class TestSimplexProjections(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(10, 3, 6, dtype=torch.float64)

    def test_project_simplex(self):
        projected = project_simplex(self.x)
        self.assertEqual(projected.shape, self.x.shape)
        self.assertTrue(bool((projected >= 0).all()))
        self.assertTrue(
            torch.allclose(projected.sum(dim=-1), torch.ones(10, 3, dtype=self.x.dtype))
        )
        self.assertTrue(torch.allclose(project_simplex(projected), projected))

    def test_project_simplex_is_closest_point(self):
        projected = project_simplex(self.x)
        other = torch.distributions.Dirichlet(torch.ones(6, dtype=self.x.dtype)).sample(
            (10, 3)
        )
        self.assertTrue(
            bool(
                (
                    (self.x - projected).norm(dim=-1)
                    <= (self.x - other).norm(dim=-1) + 1e-12
                ).all()
            )
        )

    def test_project_interior(self):
        eps = 1e-3
        interior = project_interior(self.x, eps=eps)
        self.assertTrue(bool((interior > 0.5 * eps).all()))
        self.assertTrue(
            torch.allclose(interior.sum(dim=-1), torch.ones(10, 3, dtype=self.x.dtype))
        )
        inside = torch.softmax(self.x, dim=-1).clamp_min(eps)
        inside = inside / inside.sum(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(project_interior(inside, eps=eps), inside))


if __name__ == "__main__":
    unittest.main()
