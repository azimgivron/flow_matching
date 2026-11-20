# Copyright (c) 2026 azim-givron
#
# This source code is licensed under the MIT license.
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
from torch import Tensor

from flow_matching.path.potential.potential import LogBarrierPotential


class NegativeEntropy(LogBarrierPotential):
    r"""Negative entropy :math:`\phi(x) = \sum_k x_k \log x_k`.

    Its Bregman divergence is :math:`\mathrm{KL}(x\|y)`, its metric is Fisher–Rao,
    :math:`\nabla^2\phi(x) = \mathrm{diag}(\mathbf{1} \oslash x)`, and its conditional
    paths are the e-geodesics :math:`x_t = \mathrm{softmax}((1-t)\log x_0 + t\log x_1)`.
    The Legendre map, velocity and metric have closed forms (paper, App. F.1).
    """

    def forward(self, x: Tensor) -> Tensor:
        r"""Evaluate :math:`\phi(x) = \sum_k x_k \log x_k`.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).

        Returns:
            Tensor: :math:`\phi(x)`, shape (...).
        """
        return torch.xlogy(x, x).sum(dim=-1)

    def gradient(self, x: Tensor) -> Tensor:
        r"""Evaluate :math:`\nabla\phi(x) = \log x + \mathbf{1}`.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).

        Returns:
            Tensor: :math:`\nabla\phi(x)`, shape (..., K).
        """
        return x.log() + 1.0

    def hessian(self, x: Tensor) -> Tensor:
        r"""Evaluate :math:`\nabla^2\phi(x) = \mathrm{diag}(\mathbf{1} \oslash x)`.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).

        Returns:
            Tensor: :math:`\nabla^2\phi(x)`, shape (..., K, K).
        """
        return torch.diag_embed(x.reciprocal())

    def metric(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        r"""Fisher–Rao metric :math:`g_x(u, v) = \sum_k u_k v_k / x_k`.

        Args:
            x (Tensor): base points, shape (..., K).
            u (Tensor): first tangent vectors, shape (..., K).
            v (Tensor): second tangent vectors, shape (..., K).

        Returns:
            Tensor: inner products, shape (...).
        """
        return (u * v / x).sum(dim=-1)

    def legendre_map(
        self, dual: Tensor, x_init: Optional[Tensor] = None, **solver_kwargs
    ) -> Tensor:
        r"""Closed-form Legendre map :math:`x(\gamma) = \mathrm{softmax}(\gamma)`.

        Args:
            dual (Tensor): dual coordinates :math:`\gamma`, shape (..., K).
            x_init (Optional[Tensor]): unused.
            **solver_kwargs: unused.

        Returns:
            Tensor: :math:`\mathrm{softmax}(\gamma)`, shape (..., K).
        """
        return torch.softmax(dual, dim=-1)

    def velocity(self, x: Tensor, d_dual: Tensor) -> Tensor:
        r"""Closed-form velocity :math:`\dot x = x \odot (\dot\gamma - \langle x, \dot\gamma\rangle\mathbf{1})`.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).
            d_dual (Tensor): dual velocities :math:`\dot\gamma`, shape (..., K).

        Returns:
            Tensor: tangent velocities :math:`\dot x`, shape (..., K).
        """
        return x * (d_dual - (x * d_dual).sum(dim=-1, keepdim=True))
