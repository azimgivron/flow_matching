# Copyright (c) 2026 azim-givron
#
# This source code is licensed under the MIT license.
# SPDX-License-Identifier: MIT
import unittest

import torch
from flow_matching.path.potential import KernelSmoothedEntropy, NegativeEntropy
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper, project_interior
from torch import nn


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


class MLP(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes + 1, 16, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(16, num_classes, dtype=torch.float64),
        )

    def forward(self, x, t):
        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t], dim=-1))


class HessianHead(ModelWrapper):
    # Output head of the paper, App. I.2, as a user would write it.
    def __init__(self, model, potential):
        super().__init__(model)
        self.potential = potential

    def forward(self, x, t, **extras):
        return self.potential.tangent_from_output(x, self.model(x=x, t=t, **extras))


class TestHessianLikelihood(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.num_classes = 4
        kernels = lazy_kernel_bank(self.num_classes, 1)
        self.models = [
            HessianHead(MLP(self.num_classes), potential)
            for potential in [
                NegativeEntropy(),
                KernelSmoothedEntropy(
                    kernels, torch.tensor([0.5], dtype=torch.float64)
                ),
            ]
        ]

    def test_ambient_divergence_equals_intrinsic_divergence(self):
        # Orthonormal basis of the tangent space T = {w : 1^T w = 0}.
        ones = torch.ones(self.num_classes, 1, dtype=torch.float64)
        basis, _ = torch.linalg.qr(
            torch.eye(self.num_classes, dtype=torch.float64) - ones @ ones.T / 4
        )
        basis = basis[:, : self.num_classes - 1]
        t = torch.tensor([0.4], dtype=torch.float64)
        for model in self.models:
            for x in random_interior(5, self.num_classes):
                jacobian = torch.autograd.functional.jacobian(
                    lambda z, model=model: model(z.unsqueeze(0), t).squeeze(0), x
                )
                intrinsic = torch.trace(basis.T @ jacobian @ basis)
                self.assertTrue(torch.allclose(torch.trace(jacobian), intrinsic))

    def test_sample_and_compute_likelihood(self):
        prior = torch.distributions.Dirichlet(
            torch.ones(self.num_classes, dtype=torch.float64)
        )
        x_1 = random_interior(6, self.num_classes)
        for model in self.models:
            solver = ODESolver(velocity_model=model)
            samples = solver.sample(
                x_init=random_interior(6, self.num_classes), step_size=0.05
            )
            self.assertTrue(
                torch.allclose(samples.sum(dim=-1), torch.ones(6, dtype=torch.float64))
            )
            x_0, log_p = solver.compute_likelihood(
                x_1=x_1,
                log_p0=prior.log_prob,
                step_size=0.05,
                exact_divergence=True,
            )
            self.assertEqual(log_p.shape, (6,))
            self.assertTrue(bool(torch.isfinite(log_p).all()))


if __name__ == "__main__":
    unittest.main()
