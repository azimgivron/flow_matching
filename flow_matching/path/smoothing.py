# Copyright (c) 2026 azim-givron
#
# This source code is licensed under the MIT license.
# SPDX-License-Identifier: MIT

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.distributions import Dirichlet

from flow_matching.utils.simplex import project_interior


class LabelSmoothing(ABC):
    r"""Abstract map from categorical labels to points in the open simplex.

    Simplex flows evolve in the interior of the simplex, where the Hessian geometry is
    smooth, so one-hot observations :math:`e_k` are replaced by interior targets before
    training (paper, App. H.1). Subclasses implement :meth:`smooth` and, optionally,
    :meth:`log_prob`, the smoothing density used as importance proposal in the
    categorical likelihood lower bound (paper, App. H.2).
    """

    @abstractmethod
    def smooth(self, labels: Tensor, stochastic: bool = True) -> Tensor:
        r"""Map one-hot labels into the open simplex.

        Args:
            labels (Tensor): one-hot labels, shape (..., K).
            stochastic (bool): sample the smoothing noise if True, otherwise use a deterministic representative.

        Returns:
            Tensor: points in the open simplex, shape (..., K).
        """

    def log_prob(self, x: Tensor, labels: Tensor) -> Tensor:
        r"""Log-density of the smoothing distribution :math:`q(x \mid e_k)` on the affine simplex.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).
            labels (Tensor): one-hot labels, shape (..., K).

        Raises:
            NotImplementedError: if the subclass does not provide a density.

        Returns:
            Tensor: log-densities with respect to the Lebesgue measure of the first K-1 coordinates (the convention of ``torch.distributions.Dirichlet``), shape (...).
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement log_prob.")

    def __call__(self, labels: Tensor, stochastic: bool = True) -> Tensor:
        r"""Alias of :meth:`smooth`.

        Args:
            labels (Tensor): one-hot labels, shape (..., K).
            stochastic (bool): sample the smoothing noise if True.

        Returns:
            Tensor: points in the open simplex, shape (..., K).
        """
        return self.smooth(labels, stochastic=stochastic)


class DirichletSmoothing(LabelSmoothing):
    r"""Bounded Dirichlet interior smoothing (paper, App. H.1).

    .. math::

        x = \nu e_k + (1-\nu)\left[\tau\mathbf{1} + (1 - K\tau)\zeta\right],
        \qquad \zeta \sim \mathrm{Dir}(\alpha, \ldots, \alpha), \quad \tau = m / K.

    Every coordinate satisfies :math:`x_i > (1-\nu)\tau`, so the targets stay away from the
    boundary, and :math:`x_k \geq \nu > 1/2`, so the argmax decodes the label. With
    ``stochastic=False``, :math:`\zeta` is replaced by its mean :math:`\mathbf{1}/K`.

    Args:
        interpolation_weight (float): weight :math:`\nu \in (0.5, 1)` of the vertex.
        dirichlet_alpha (float): symmetric Dirichlet concentration :math:`\alpha > 0`.
        boundary_mass (float): total boundary mass :math:`m = K\tau \in (0, 1)`.
        eps (float): lower bound of the final interior projection.

    Raises:
        ValueError: if a parameter is out of range.
    """

    def __init__(
        self,
        interpolation_weight: float = 0.9,
        dirichlet_alpha: float = 100.0,
        boundary_mass: float = 0.2,
        eps: float = 1e-4,
    ) -> None:
        if not 0.5 < interpolation_weight < 1.0:
            raise ValueError("interpolation_weight must be in (0.5, 1.0).")
        if not math.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0.0:
            raise ValueError("dirichlet_alpha must be positive and finite.")
        if not 0.0 < boundary_mass < 1.0:
            raise ValueError("boundary_mass must be in (0, 1).")
        self.interpolation_weight = interpolation_weight
        self.dirichlet_alpha = dirichlet_alpha
        self.boundary_mass = boundary_mass
        self.eps = eps

    @torch.no_grad()
    def smooth(self, labels: Tensor, stochastic: bool = True) -> Tensor:
        r"""Map one-hot labels into the open simplex.

        Args:
            labels (Tensor): one-hot labels, shape (..., K).
            stochastic (bool): sample :math:`\zeta` if True, otherwise use its mean.

        Returns:
            Tensor: points in the open simplex, shape (..., K).
        """
        num_classes = labels.shape[-1]
        if stochastic:
            concentration = labels.new_full((num_classes,), self.dirichlet_alpha)
            noise = Dirichlet(concentration).sample(labels.shape[:-1])
        else:
            noise = torch.full_like(labels, 1.0 / num_classes)
        tau = self.boundary_mass / num_classes
        bounded_noise = tau + (1.0 - self.boundary_mass) * noise
        x = (
            self.interpolation_weight * labels
            + (1.0 - self.interpolation_weight) * bounded_noise
        )
        return project_interior(x, eps=self.eps)

    def log_prob(self, x: Tensor, labels: Tensor) -> Tensor:
        r"""Log-density :math:`\log q(x \mid e_k) = \log\mathrm{Dir}(\zeta(x); \alpha) - (K-1)\log b` (paper, App. H.2).

        Here :math:`b = (1-\nu)(1-K\tau)` and :math:`\zeta(x) = (x - \nu e_k - (1-\nu)\tau\mathbf{1})/b`.

        Args:
            x (Tensor): points in the open simplex, shape (..., K).
            labels (Tensor): one-hot labels, shape (..., K).

        Raises:
            ValueError: if ``x`` lies outside the support of :math:`q(\cdot \mid e_k)`.

        Returns:
            Tensor: log-densities with respect to the Lebesgue measure of the first K-1 coordinates (the convention of ``torch.distributions.Dirichlet``), shape (...).
        """
        num_classes = x.shape[-1]
        tau = self.boundary_mass / num_classes
        scale = (1.0 - self.interpolation_weight) * (1.0 - self.boundary_mass)
        zeta = (
            x
            - self.interpolation_weight * labels
            - (1.0 - self.interpolation_weight) * tau
        ) / scale
        if torch.any(zeta <= 0.0):
            raise ValueError("x lies outside the support of the smoothing density.")
        # Absorb round-off in the inversion of the narrow interpolation interval.
        zeta = zeta / zeta.sum(dim=-1, keepdim=True)
        concentration = x.new_full((num_classes,), self.dirichlet_alpha)
        return Dirichlet(concentration).log_prob(zeta) - (num_classes - 1) * math.log(
            scale
        )
