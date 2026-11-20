# Copyright (c) 2026 azim-givron
#
# This source code is licensed under the MIT license.
# SPDX-License-Identifier: MIT
import unittest

import torch
from flow_matching.path.potential import (
    additive_step,
    BregmanPotential,
    KernelSmoothedEntropy,
    multiplicative_step,
    NegativeEntropy,
    newton_lagrange,
)
from flow_matching.utils import project_interior


def lazy_kernel_bank(num_classes: int, num_kernels: int, alpha: float = 0.4):
    affinity = torch.rand(num_classes, num_classes, dtype=torch.float64)
    affinity = (affinity + affinity.T) / 2
    affinity.fill_diagonal_(0.0)
    jump = affinity / affinity.sum(dim=0, keepdim=True)
    lazy = (1 - alpha) * torch.eye(num_classes, dtype=torch.float64) + alpha * jump
    powers = [lazy]
    for _ in range(num_kernels - 1):
        powers.append(powers[-1] @ lazy)
    return torch.stack(powers)


def random_interior(*shape: int) -> torch.Tensor:
    concentration = torch.ones(shape[-1], dtype=torch.float64)
    return project_interior(
        torch.distributions.Dirichlet(concentration).sample(shape[:-1])
    )


class AdditiveNegativeEntropy(NegativeEntropy):
    @staticmethod
    def legendre_step(x, direction, alpha):
        return additive_step(x, direction, alpha)


class TestPotentials(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.num_classes = 5
        self.x = random_interior(8, 3, self.num_classes)
        self.kernels = lazy_kernel_bank(self.num_classes, 2)
        self.lambdas = torch.tensor([0.5, 0.3], dtype=torch.float64)
        self.potentials = [
            NegativeEntropy(),
            KernelSmoothedEntropy(self.kernels, self.lambdas),
        ]

    def test_gradient_matches_autograd(self):
        for potential in self.potentials:
            x = self.x.clone().requires_grad_(True)
            expected = torch.autograd.grad(potential(x).sum(), x)[0]
            self.assertTrue(torch.allclose(potential.gradient(self.x), expected))

    def test_hessian_matches_autograd(self):
        for potential in self.potentials:
            point = self.x[0, 0]
            expected = torch.autograd.functional.hessian(potential, point)
            self.assertTrue(torch.allclose(potential.hessian(point), expected))

    def test_kernel_smoothed_entropy_with_zero_lambdas_is_negative_entropy(self):
        kse = KernelSmoothedEntropy(self.kernels, torch.zeros(2, dtype=torch.float64))
        ne = NegativeEntropy()
        self.assertTrue(torch.allclose(kse(self.x), ne(self.x)))
        self.assertTrue(torch.allclose(kse.gradient(self.x), ne.gradient(self.x)))
        self.assertTrue(torch.allclose(kse.hessian(self.x), ne.hessian(self.x)))

    def test_kernel_smoothed_entropy_validation(self):
        with self.assertRaises(ValueError):
            KernelSmoothedEntropy(self.kernels, torch.tensor([0.8, 0.3]))
        with self.assertRaises(ValueError):
            KernelSmoothedEntropy(self.kernels, torch.tensor([0.8, -0.1]))
        with self.assertRaises(ValueError):
            KernelSmoothedEntropy(2 * self.kernels, self.lambdas)
        with self.assertRaises(ValueError):
            KernelSmoothedEntropy(self.kernels[0], self.lambdas)

    def test_bregman_divergence_is_weighted_kl(self):
        kse = KernelSmoothedEntropy(self.kernels, self.lambdas)
        y = random_interior(8, 3, self.num_classes)

        def kl(a, b):
            return (a * (a / b).log()).sum(dim=-1)

        expected = (1 - self.lambdas.sum()) * kl(self.x, y)
        for kernel, weight in zip(self.kernels, self.lambdas):
            expected = expected + weight * kl(self.x @ kernel.T, y @ kernel.T)
        self.assertTrue(torch.allclose(kse.bregman_divergence(self.x, y), expected))

    def test_metric(self):
        u = torch.randn_like(self.x)
        v = torch.randn_like(self.x)
        for potential in self.potentials:
            expected = torch.einsum(
                "...i,...ij,...j->...", u, potential.hessian(self.x), v
            )
            self.assertTrue(torch.allclose(potential.metric(self.x, u, v), expected))

    def test_rectangular_kernel(self):
        # A many-to-one kernel merging the first two and the last three categories.
        kernel = torch.zeros(1, 2, self.num_classes, dtype=torch.float64)
        kernel[0, 0, :2] = 1.0
        kernel[0, 1, 2:] = 1.0
        kse = KernelSmoothedEntropy(kernel, torch.tensor([0.6], dtype=torch.float64))
        x = self.x.clone().requires_grad_(True)
        expected = torch.autograd.grad(kse(x).sum(), x)[0]
        self.assertTrue(torch.allclose(kse.gradient(self.x), expected))
        self.assertTrue(
            torch.allclose(
                kse.legendre_map(kse.dual(self.x), tol=1e-7), self.x, atol=1e-7
            )
        )

    def test_legendre_map_inverts_gradient(self):
        for potential in self.potentials:
            recovered = potential.legendre_map(potential.dual(self.x), tol=1e-7)
            self.assertTrue(torch.allclose(recovered, self.x, atol=1e-7))

    def test_legendre_map_is_invariant_to_dual_shift(self):
        kse = self.potentials[1]
        dual = torch.randn_like(self.x)
        shifted = dual + torch.randn(*dual.shape[:-1], 1, dtype=dual.dtype)
        self.assertTrue(
            torch.allclose(
                kse.legendre_map(dual, tol=1e-7),
                kse.legendre_map(shifted, tol=1e-7),
                atol=1e-7,
            )
        )

    def test_generic_newton_matches_softmax_for_negative_entropy(self):
        dual = torch.randn_like(self.x)
        expected = torch.softmax(dual, dim=-1)
        for potential in [NegativeEntropy(), AdditiveNegativeEntropy()]:
            generic = BregmanPotential.legendre_map(potential, dual, tol=1e-10)
            self.assertTrue(torch.allclose(generic, expected, atol=1e-7))

    def test_multiplicative_step_is_exact_for_negative_entropy(self):
        ne = NegativeEntropy()
        dual = torch.randn_like(self.x)
        _, stats = newton_lagrange(
            torch.full_like(self.x, 1.0 / self.num_classes),
            dual,
            objective=ne.objective,
            gradient=ne.gradient,
            hessian=ne.hessian,
            step=multiplicative_step,
            tol=1e-10,
            return_stats=True,
        )
        self.assertEqual(stats.iterations, 1)

    def test_generic_velocity_matches_closed_form(self):
        ne = NegativeEntropy()
        d_dual = torch.randn_like(self.x)
        generic = BregmanPotential.velocity(ne, self.x, d_dual)
        self.assertTrue(torch.allclose(generic, ne.velocity(self.x, d_dual)))
        self.assertTrue(
            torch.allclose(generic.sum(dim=-1), torch.zeros(8, 3, dtype=torch.float64))
        )


if __name__ == "__main__":
    unittest.main()
