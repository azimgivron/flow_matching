# Copyright (c) 2026 azim-givron
#
# This source code is licensed under the MIT license.
# SPDX-License-Identifier: MIT

import torch
from torch import Tensor


def project_simplex(x: Tensor) -> Tensor:
    r"""Euclidean projection onto the probability simplex along the last axis.

    Uses the sort-based algorithm: :math:`\Pi(x) = \max(x - \theta\mathbf{1}, 0)`, with
    :math:`\theta` chosen so that the result sums to one.

    Args:
        x (Tensor): points, shape (..., K).

    Returns:
        Tensor: projected points on the closed simplex, shape (..., K).
    """
    flat = x.reshape(-1, x.shape[-1])
    sorted_x, _ = torch.sort(flat, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_x, dim=-1) - 1.0
    ranks = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
    threshold = cumsum / ranks
    support = (sorted_x > threshold).sum(dim=-1, keepdim=True)
    theta = torch.gather(threshold, dim=-1, index=support - 1)
    return torch.clamp(flat - theta, min=0.0).reshape(x.shape)


def project_interior(x: Tensor, eps: float = 1e-4) -> Tensor:
    r"""Project onto the open simplex: Euclidean projection, clip below at ``eps``, renormalize.

    Args:
        x (Tensor): points, shape (..., K).
        eps (float): lower bound applied to every coordinate before renormalization.

    Returns:
        Tensor: points in the open simplex, shape (..., K).
    """
    interior = project_simplex(x).clamp_min(eps)
    return interior / interior.sum(dim=-1, keepdim=True)
