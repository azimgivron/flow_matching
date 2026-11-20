# Copyright (c) 2026 azim-givron
#
# This source code is licensed under the MIT license.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn, Tensor

from flow_matching.path.potential.solvers import (
    additive_step,
    multiplicative_step,
    newton_lagrange,
    tangent_solve,
)


class BregmanPotential(nn.Module, ABC):
    r"""Abstract strictly convex potential :math:`\phi` (Bregman generator) on the open simplex.

    A potential defines every ingredient of Hessian flow matching:

    - the dual coordinates :math:`\gamma = \nabla\phi(x)`, in which conditional paths are affine;
    - the simplex-constrained Legendre map :math:`x(\gamma)`, which maps them back to the simplex;
    - the path velocity :math:`\dot x = H(x)\dot\gamma`, with :math:`H` the inverse of the metric on the tangent space;
    - the regression metric :math:`\|u\|^2_{g_x} = u^\top\nabla^2\phi(x)u`;
    - the network output head mapping a raw output :math:`r \in \mathbb{R}^K` to a tangent velocity.

    Subclasses implement :meth:`forward`, :meth:`gradient` and :meth:`hessian`. The
    remaining methods have generic numerical defaults (Newton–Lagrange for the Legendre
    map, a Schur-complement solve for the velocity) that subclasses may override with
    closed forms. All methods act on the last axis and broadcast over the leading ones,
    so a sequence of :math:`L` sites with shape (batch, L, K) uses the separable
    generator :math:`\sum_\ell \phi(x^{(\ell)})`.

    A valid potential must be strictly convex and steep at the simplex boundary
    (paper, Assumption 1).

    Example:

    .. code-block:: python

        class WeightedEntropy(LogBarrierPotential):
            # phi(x) = sum_k w_k x_k log x_k, with w_k > 0.
            def __init__(self, weights: Tensor):
                super().__init__()
                self.register_buffer("weights", weights)

            def forward(self, x):
                return (self.weights * torch.xlogy(x, x)).sum(dim=-1)

            def gradient(self, x):
                return self.weights * (x.log() + 1.0)

            def hessian(self, x):
                return torch.diag_embed(self.weights / x)
    """

    @staticmethod
    def legendre_step(x: Tensor, direction: Tensor, alpha: Tensor) -> Tensor:
        r"""Step rule of the Newton iterations of :meth:`legendre_map`: :math:`x + \alpha d`.

        Args:
            x (Tensor): current iterate, shape (..., K).
            direction (Tensor): Newton direction, shape (..., K).
            alpha (Tensor): step sizes, shape (...).

        Returns:
            Tensor: trial iterate, shape (..., K).
        """
        return additive_step(x, direction, alpha)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        r"""Evaluate the potential :math:`\phi(x)`.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).

        Returns:
            Tensor: :math:`\phi(x)`, shape (...).
        """

    @abstractmethod
    def gradient(self, x: Tensor) -> Tensor:
        r"""Evaluate the gradient :math:`\nabla\phi(x)`.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).

        Returns:
            Tensor: :math:`\nabla\phi(x)`, shape (..., K).
        """

    @abstractmethod
    def hessian(self, x: Tensor) -> Tensor:
        r"""Evaluate the Hessian :math:`G(x) = \nabla^2\phi(x)`.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).

        Returns:
            Tensor: :math:`\nabla^2\phi(x)`, shape (..., K, K).
        """

    def dual(self, x: Tensor) -> Tensor:
        r"""Map primal points to dual coordinates :math:`\gamma = \nabla\phi(x)`.

        Dual coordinates are defined up to adding a multiple of :math:`\mathbf{1}`.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).

        Returns:
            Tensor: dual coordinates, shape (..., K).
        """
        return self.gradient(x)

    def bregman_divergence(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Evaluate :math:`D_\phi(x, y) = \phi(x) - \phi(y) - \langle\nabla\phi(y), x - y\rangle`.

        Args:
            x (Tensor): first argument, shape (..., K).
            y (Tensor): second argument, shape (..., K).

        Returns:
            Tensor: divergences, shape (...).
        """
        return self(x) - self(y) - (self.gradient(y) * (x - y)).sum(dim=-1)

    def metric(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        r"""Evaluate the Hessian metric :math:`g_x(u, v) = u^\top\nabla^2\phi(x)v`.

        Args:
            x (Tensor): base points, shape (..., K).
            u (Tensor): first tangent vectors, shape (..., K).
            v (Tensor): second tangent vectors, shape (..., K).

        Returns:
            Tensor: inner products, shape (...).
        """
        hv = (self.hessian(x) @ v.unsqueeze(-1)).squeeze(-1)
        return (u * hv).sum(dim=-1)

    def objective(self, x: Tensor, dual: Tensor) -> Tensor:
        r"""Evaluate the Legendre objective :math:`\phi(x) - \langle\gamma, x\rangle`.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).
            dual (Tensor): dual coordinates :math:`\gamma`, shape (..., K).

        Returns:
            Tensor: objective values, shape (...).
        """
        return self(x) - (dual * x).sum(dim=-1)

    def legendre_map(
        self, dual: Tensor, x_init: Optional[Tensor] = None, **solver_kwargs
    ) -> Tensor:
        r"""Simplex-constrained Legendre map :math:`x(\gamma) = \arg\max_{x} \langle\gamma, x\rangle - \phi(x)` (paper, Eq. 5).

        The generic implementation runs damped Newton–Lagrange iterations (paper,
        App. I.1) with the step rule :meth:`legendre_step`, started at ``x_init`` or,
        by default, at :math:`\mathrm{softmax}(\gamma)`, the negative-entropy solution.

        Args:
            dual (Tensor): dual coordinates :math:`\gamma`, shape (..., K).
            x_init (Optional[Tensor]): initial iterate in the open simplex, shape (..., K).
            **solver_kwargs: keyword arguments forwarded to :func:`~flow_matching.path.potential.solvers.newton_lagrange` (e.g. ``tol``, ``max_iters``).

        Returns:
            Tensor: :math:`x(\gamma)`, shape (..., K).
        """
        if x_init is None:
            x_init = torch.softmax(dual, dim=-1)
        return newton_lagrange(
            x_init,
            dual,
            objective=self.objective,
            gradient=self.gradient,
            hessian=self.hessian,
            step=self.legendre_step,
            **solver_kwargs,
        )

    def velocity(self, x: Tensor, d_dual: Tensor) -> Tensor:
        r"""Push a dual velocity to a tangent velocity, :math:`\dot x = H(x)\dot\gamma` (paper, Eq. 10).

        Solves :math:`\nabla^2\phi(x)\dot x + \dot\eta\mathbf{1} = \dot\gamma` with
        :math:`\mathbf{1}^\top\dot x = 0`.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).
            d_dual (Tensor): dual velocities :math:`\dot\gamma`, shape (..., K).

        Returns:
            Tensor: tangent velocities :math:`\dot x`, shape (..., K).
        """
        return tangent_solve(self.hessian(x), d_dual)

    def tangent_from_output(self, x: Tensor, output: Tensor) -> Tensor:
        r"""Network output head: map a raw output :math:`r` to a tangent vector.

        The default is the Euclidean projection :math:`\Pi_{\mathcal{T}} r = r - \bar r\mathbf{1}`,
        which is the constrained inverse Hessian of potentials without a logarithmic
        barrier, such as :math:`\tfrac12\|x\|^2` (paper, App. I.2).

        Args:
            x (Tensor): points in the open simplex, shape (..., K).
            output (Tensor): raw network output :math:`r`, shape (..., K).

        Returns:
            Tensor: tangent vectors, shape (..., K).
        """
        return output - output.mean(dim=-1, keepdim=True)


class LogBarrierPotential(BregmanPotential, ABC):
    r"""Abstract potential with a logarithmic barrier at the simplex boundary.

    Compared to :class:`BregmanPotential`, it changes two defaults (paper, App. I):

    - the Newton steps of the Legendre map are taken in log coordinates, which keeps
      every iterate in the open simplex (:func:`~flow_matching.path.potential.solvers.multiplicative_step`);
    - the output head is :math:`v = S(x)r = x \odot (r - \langle x, r\rangle\mathbf{1})`,
      with :math:`S(x) = \mathrm{diag}(x) - xx^\top` the constrained inverse Hessian of
      negative entropy (paper, Eq. 29). The loss then weights the raw-output error by
      :math:`x_k` instead of :math:`1/x_k`. For negative entropy the optimal raw output is
      :math:`r^\star_t(x) = (\mathbb{E}[\log x_1 \mid x_t = x] - \log x)/(1-t)`, a
      posterior-mean (denoising) target (paper, App. I.2, Property 4).
    """

    @staticmethod
    def legendre_step(x: Tensor, direction: Tensor, alpha: Tensor) -> Tensor:
        r"""Step rule in log coordinates, :math:`\mathrm{softmax}(\log x + \alpha d \oslash x)` (paper, Eq. 28).

        Args:
            x (Tensor): current positive iterate, shape (..., K).
            direction (Tensor): tangent Newton direction, shape (..., K).
            alpha (Tensor): step sizes, shape (...).

        Returns:
            Tensor: trial iterate in the open simplex, shape (..., K).
        """
        return multiplicative_step(x, direction, alpha)

    def tangent_from_output(self, x: Tensor, output: Tensor) -> Tensor:
        r"""Output head :math:`v = x \odot (r - \langle x, r\rangle / \langle x, \mathbf{1}\rangle)` (paper, Eq. 29).

        Args:
            x (Tensor): points in the open simplex, shape (..., K).
            output (Tensor): raw network output :math:`r`, shape (..., K).

        Returns:
            Tensor: tangent vectors, shape (..., K).
        """
        weighted_mean = (x * output).sum(dim=-1, keepdim=True) / x.sum(
            dim=-1, keepdim=True
        )
        return x * (output - weighted_mean)
