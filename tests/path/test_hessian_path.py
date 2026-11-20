# Copyright (c) 2026 azim-givron
#
# This source code is licensed under the MIT license.
# SPDX-License-Identifier: MIT
import unittest

import torch
from flow_matching.path import DirichletSmoothing, HessianProbPath
from flow_matching.path.potential import KernelSmoothedEntropy, NegativeEntropy
from flow_matching.path.scheduler import CondOTScheduler, PolynomialConvexScheduler

from .test_potential import lazy_kernel_bank, random_interior


class TestHessianProbPath(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.num_classes = 4
        self.batch_size = 16
        kernels = lazy_kernel_bank(self.num_classes, 2)
        lambdas = torch.tensor([0.5, 0.3], dtype=torch.float64)
        self.paths = [
            HessianProbPath(
                CondOTScheduler(), NegativeEntropy(), solver_kwargs={"tol": 1e-7}
            ),
            HessianProbPath(
                CondOTScheduler(),
                KernelSmoothedEntropy(kernels, lambdas),
                solver_kwargs={"tol": 1e-7},
            ),
        ]
        labels = torch.nn.functional.one_hot(
            torch.randint(0, self.num_classes, (self.batch_size, 3)), self.num_classes
        ).to(torch.float64)
        self.x_0 = random_interior(self.batch_size, 3, self.num_classes)
        self.x_1 = DirichletSmoothing()(labels)
        self.t = torch.rand(self.batch_size, dtype=torch.float64) * 0.8 + 0.1

    def test_sample_shapes_and_passthrough(self):
        for path in self.paths:
            sample = path.sample(self.x_0, self.x_1, self.t)
            self.assertEqual(sample.x_t.shape, self.x_0.shape)
            self.assertEqual(sample.dx_t.shape, self.x_0.shape)
            self.assertTrue(torch.equal(sample.x_0, self.x_0))
            self.assertTrue(torch.equal(sample.x_1, self.x_1))
            self.assertTrue(torch.equal(sample.t, self.t))

    def test_endpoints(self):
        for path in self.paths:
            zeros = torch.zeros(self.batch_size, dtype=torch.float64)
            start = path.sample(self.x_0, self.x_1, zeros).x_t
            end = path.sample(self.x_0, self.x_1, torch.ones_like(zeros)).x_t
            self.assertTrue(torch.allclose(start, self.x_0, atol=1e-7))
            self.assertTrue(torch.allclose(end, self.x_1, atol=1e-7))

    def test_path_stays_in_simplex_and_velocity_is_tangent(self):
        for path in self.paths:
            sample = path.sample(self.x_0, self.x_1, self.t)
            self.assertTrue(bool((sample.x_t > 0).all()))
            ones = torch.ones(self.batch_size, 3, dtype=torch.float64)
            self.assertTrue(torch.allclose(sample.x_t.sum(dim=-1), ones))
            self.assertTrue(
                torch.allclose(sample.dx_t.sum(dim=-1), torch.zeros_like(ones))
            )

    def test_velocity_matches_finite_difference(self):
        h = 1e-3
        schedulers = [CondOTScheduler(), PolynomialConvexScheduler(n=2.0)]
        for path in self.paths:
            for scheduler in schedulers:
                path = HessianProbPath(
                    scheduler, path.potential, solver_kwargs={"tol": 1e-7}
                )
                dx_t = path.sample(self.x_0, self.x_1, self.t).dx_t
                forward = path.sample(self.x_0, self.x_1, self.t + h).x_t
                backward = path.sample(self.x_0, self.x_1, self.t - h).x_t
                self.assertTrue(
                    torch.allclose(dx_t, (forward - backward) / (2 * h), atol=1e-4)
                )

    def test_dtype_and_flat_input(self):
        for path in self.paths:
            x_0, x_1 = self.x_0[:, 0].float(), self.x_1[:, 0].float()
            sample = path.sample(x_0, x_1, self.t.float())
            self.assertEqual(sample.x_t.dtype, torch.float32)
            self.assertEqual(sample.dx_t.shape, x_0.shape)


class TestOutputHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.num_classes = 5
        self.x = random_interior(32, self.num_classes)
        self.delta = torch.randn_like(self.x)
        kernels = lazy_kernel_bank(self.num_classes, 2)
        self.lambdas = torch.tensor([0.5, 0.3], dtype=torch.float64)
        self.kse = KernelSmoothedEntropy(kernels, self.lambdas)

    def variance(self):
        mean = (self.x * self.delta).sum(dim=-1, keepdim=True)
        return (self.x * (self.delta - mean).square()).sum(dim=-1)

    def test_negative_entropy_weighting_is_variance(self):
        # Paper, App. I.2, Property 2.
        ne = NegativeEntropy()
        error = ne.tangent_from_output(self.x, self.delta)
        self.assertTrue(
            torch.allclose(ne.metric(self.x, error, error), self.variance())
        )

    def test_kernel_smoothed_entropy_weighting_is_bounded(self):
        # Paper, App. I.2, Property 3.
        error = self.kse.tangent_from_output(self.x, self.delta)
        loss = self.kse.metric(self.x, error, error)
        lambda_0 = 1 - self.lambdas.sum()
        self.assertTrue(bool((loss <= self.variance() + 1e-12).all()))
        self.assertTrue(bool((loss >= lambda_0 * self.variance() - 1e-12).all()))

    def test_head_is_tangent_everywhere(self):
        # Tangent for any base point, not only on the simplex: the ambient divergence
        # of the velocity then equals the intrinsic one.
        x = torch.randn(7, 3, self.num_classes, dtype=torch.float64)
        output = torch.randn_like(x)
        for potential in [NegativeEntropy(), self.kse]:
            velocity = potential.tangent_from_output(x, output)
            self.assertTrue(
                torch.allclose(velocity.sum(dim=-1), torch.zeros(7, 3, dtype=x.dtype))
            )


if __name__ == "__main__":
    unittest.main()
