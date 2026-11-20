# Copyright (c) 2026 azim-givron
#
# This source code is licensed under the MIT license.
# SPDX-License-Identifier: MIT

from math import isclose

import torch
from torch import Tensor

from flow_matching.path.potential.potential import LogBarrierPotential


class KernelSmoothedEntropy(LogBarrierPotential):
    r"""Kernel-smoothed entropy (paper, Proposition 4).

    .. math::

        \phi(x) = \lambda_0 \sum_i x_i\log x_i + \sum_{j=1}^{J} \lambda_j \sum_o (M_j x)_o \log (M_j x)_o,
        \qquad \lambda_0 = 1 - \sum_{j} \lambda_j.

    Each kernel :math:`M_j \in \mathbb{R}^{O\times K}` is column stochastic, so :math:`M_j x`
    is :math:`x` passed through a substitution channel and is again a distribution. The
    divergence is :math:`D_\phi(x, y) = \sum_j \lambda_j \mathrm{KL}(M_j x\|M_j y)` (with
    :math:`M_0 = I`), which discounts discrepancies between categories that the kernels
    frequently interchange. The identity view :math:`\lambda_0 > 0` keeps the generator
    strictly convex and steep at the boundary.

    The Legendre map has no closed form and uses the generic Newton solver of
    :class:`~flow_matching.path.potential.BregmanPotential`.

    Args:
        kernels (Tensor): column-stochastic kernels without zero rows, shape (J, O, K).
        lambdas (Tensor): non-negative weights :math:`\lambda_j`, shape (J,), summing to at most one.
        atol (float): absolute tolerance of the stochasticity and weight checks.
        rtol (float): relative tolerance of the weight-sum check.

    Raises:
        ValueError: if ``kernels`` or ``lambdas`` have invalid shapes or values.
    """

    def __init__(
        self, kernels: Tensor, lambdas: Tensor, atol: float = 1e-6, rtol: float = 1e-5
    ) -> None:
        super().__init__()
        if kernels.ndim != 3:
            raise ValueError(
                "kernels must have shape (num_kernels, out_dim, dim), "
                f"got {tuple(kernels.shape)}."
            )
        if lambdas.ndim != 1:
            raise ValueError(
                f"lambdas must have shape (num_kernels,), got {tuple(lambdas.shape)}."
            )
        if lambdas.shape[0] != kernels.shape[0]:
            raise ValueError(
                "lambdas and kernels must have the same number of kernels, got "
                f"{lambdas.shape[0]} and {kernels.shape[0]}."
            )
        if (lambdas < 0).any():
            raise ValueError("lambdas must be non-negative.")
        lambda_sum = lambdas.sum().item()
        if lambda_sum > 1.0 and not isclose(
            lambda_sum, 1.0, abs_tol=atol, rel_tol=rtol
        ):
            raise ValueError(
                "lambdas must sum to at most 1 because lambda_0 is implicit. "
                f"Got sum {lambda_sum}."
            )
        if (kernels < 0).any():
            raise ValueError("kernels must be non-negative.")
        column_sums = kernels.sum(dim=1)
        if not torch.allclose(
            column_sums, torch.ones_like(column_sums), atol=max(atol, 1e-5)
        ):
            raise ValueError("kernels must be column stochastic.")
        if (kernels.sum(dim=2) <= 0).any():
            raise ValueError("kernels must not have zero rows.")

        self.register_buffer("kernels", kernels.detach().clone())
        self.register_buffer("lambdas", lambdas.detach().clone())

    def _weights(self, x: Tensor) -> tuple:
        r"""Return the kernels, kernel weights and identity weight in the dtype and device of ``x``.

        Args:
            x (Tensor): reference tensor.

        Returns:
            tuple: ``(kernels, lambdas, lambda_0)``.
        """
        kernels = self.kernels.to(device=x.device, dtype=x.dtype)
        lambdas = self.lambdas.to(device=x.device, dtype=x.dtype)
        return kernels, lambdas, (1.0 - lambdas.sum()).clamp_min(0.0)

    def _views(self, x: Tensor, kernels: Tensor) -> Tensor:
        r"""Compute the smoothed views :math:`M_j x`.

        Args:
            x (Tensor): points, shape (..., K).
            kernels (Tensor): kernels, shape (J, O, K).

        Raises:
            ValueError: if the kernel input dimension does not match ``x``.

        Returns:
            Tensor: smoothed views, shape (..., J, O).
        """
        if kernels.shape[-1] != x.shape[-1]:
            raise ValueError(
                f"Kernel dimension {kernels.shape[-1]} must match simplex dimension {x.shape[-1]}."
            )
        return torch.einsum("jok,...k->...jo", kernels, x)

    def forward(self, x: Tensor) -> Tensor:
        r"""Evaluate :math:`\phi(x)`.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).

        Returns:
            Tensor: :math:`\phi(x)`, shape (...).
        """
        kernels, lambdas, lambda_0 = self._weights(x)
        mx = self._views(x, kernels)
        smoothed = (torch.xlogy(mx, mx).sum(dim=-1) * lambdas).sum(dim=-1)
        return lambda_0 * torch.xlogy(x, x).sum(dim=-1) + smoothed

    def gradient(self, x: Tensor) -> Tensor:
        r"""Evaluate :math:`\nabla\phi(x) = \lambda_0(\log x + \mathbf{1}) + \sum_j \lambda_j M_j^\top(\log M_j x + \mathbf{1})`.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).

        Returns:
            Tensor: :math:`\nabla\phi(x)`, shape (..., K).
        """
        kernels, lambdas, lambda_0 = self._weights(x)
        mx = self._views(x, kernels)
        smoothed = torch.einsum("jok,...jo,j->...k", kernels, mx.log() + 1.0, lambdas)
        return lambda_0 * (x.log() + 1.0) + smoothed

    def hessian(self, x: Tensor) -> Tensor:
        r"""Evaluate :math:`\nabla^2\phi(x) = \lambda_0\mathrm{diag}(\mathbf{1}\oslash x) + \sum_j \lambda_j M_j^\top\mathrm{diag}(\mathbf{1}\oslash M_j x)M_j`.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).

        Returns:
            Tensor: :math:`\nabla^2\phi(x)`, shape (..., K, K).
        """
        kernels, lambdas, lambda_0 = self._weights(x)
        mx = self._views(x, kernels)
        smoothed = torch.einsum(
            "joi,...jo,jok,j->...ik", kernels, mx.reciprocal(), kernels, lambdas
        )
        return lambda_0 * torch.diag_embed(x.reciprocal()) + smoothed
