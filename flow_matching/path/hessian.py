# Copyright (c) 2026 azim-givron
#
# This source code is licensed under the MIT license.
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from flow_matching.path.affine import AffineProbPath
from flow_matching.path.path import ProbPath
from flow_matching.path.path_sample import PathSample
from flow_matching.path.potential import BregmanPotential
from flow_matching.path.scheduler.scheduler import Scheduler


class HessianProbPath(ProbPath):
    r"""Hessian flow matching path on the open simplex (Givron et al., Hessian Flow Matching).

    Given a potential :math:`\phi` (:class:`~flow_matching.path.potential.BregmanPotential`),
    the endpoints are interpolated affinely in dual coordinates and mapped back to the
    simplex by the constrained Legendre map:

    .. math::

        \gamma_t = \sigma_t\nabla\phi(X_0) + \alpha_t\nabla\phi(X_1), \qquad X_t = x(\gamma_t), \qquad
        \dot X_t = H(X_t)\dot\gamma_t,

    where :math:`(\alpha_t, \sigma_t)` is given by the scheduler and :math:`H` is the inverse of
    the metric on the tangent space. With :class:`~flow_matching.path.scheduler.CondOTScheduler`,
    :math:`X_t` is the Bregman barycenter :math:`\arg\min_x (1-t)D_\phi(x, X_0) + tD_\phi(x, X_1)`.
    For negative entropy this is the e-geodesic.

    The dual interpolation reuses :class:`~flow_matching.path.AffineProbPath`. Dual
    coordinates, the Legendre map and the velocity are computed in ``compute_dtype`` with
    autocast disabled, then cast back to the input dtype (paper, App. I.3).

    Inputs have shape (batch_size, ..., K): the last axis is the simplex, and the other axes
    are independent sites (the generator is separable over sites). Endpoints must lie in the
    open simplex: categorical data are first mapped to the interior, e.g. with
    :class:`~flow_matching.path.DirichletSmoothing`.

    The network is trained with the Hessian metric :math:`\|v - \dot X_t\|^2_{\nabla^2\phi(X_t)}`
    (``potential.metric``). Its raw output :math:`r` is mapped to a tangent velocity by
    ``potential.tangent_from_output`` (paper, App. I.2). That velocity is tangent everywhere,
    so :class:`~flow_matching.solver.ODESolver` can be used for sampling and for
    ``compute_likelihood``.

    Example:

    .. code-block:: python

        path = HessianProbPath(scheduler=CondOTScheduler(), potential=NegativeEntropy())
        smoothing = DirichletSmoothing()

        for labels in dataset:  # one-hot, shape (batch_size, L, K)
            x_1 = smoothing(labels)
            x_0 = project_interior(Dirichlet(torch.ones(K)).sample(x_1.shape[:-1]))
            t = torch.rand(x_1.shape[0])
            path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)

            # Output head (paper, App. I.2), then Hessian-metric regression.
            v = path.potential.tangent_from_output(path_sample.x_t, my_model(path_sample.x_t, t))
            diff = v - path_sample.dx_t
            loss = path.potential.metric(path_sample.x_t, diff, diff).sum(dim=-1).mean()
            loss.backward()

    Args:
        scheduler (Scheduler): scheduler of the dual interpolation.
        potential (BregmanPotential): the convex potential :math:`\phi`.
        compute_dtype (torch.dtype): dtype of the target computation.
        solver_kwargs (Optional[Dict[str, Any]]): keyword arguments forwarded to ``potential.legendre_map``. Defaults to ``{"tol": 1e-5}``, as in the paper (App. I.1).
    """

    def __init__(
        self,
        scheduler: Scheduler,
        potential: BregmanPotential,
        compute_dtype: torch.dtype = torch.float64,
        solver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.scheduler = scheduler
        self.potential = potential
        self.compute_dtype = compute_dtype
        self.solver_kwargs = {"tol": 1e-5} if solver_kwargs is None else solver_kwargs
        self.dual_path = AffineProbPath(scheduler=scheduler)

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        r"""Sample the Hessian path.

        | given :math:`(X_0, X_1)` in the open simplex and :math:`t`,
        | returns :math:`X_0, X_1, X_t = x(\gamma_t)` and :math:`\dot X_t = H(X_t)\dot\gamma_t`.

        Args:
            x_0 (Tensor): source points in the open simplex, shape (batch_size, ..., K).
            x_1 (Tensor): target points in the open simplex, shape (batch_size, ..., K).
            t (Tensor): times in [0, 1], shape (batch_size).

        Returns:
            PathSample: a conditional sample at :math:`X_t \sim p_t`.
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)
        dtype = x_1.dtype
        with torch.autocast(device_type=x_1.device.type, enabled=False):
            dual_sample = self.dual_path.sample(
                x_0=self.potential.dual(x_0.to(self.compute_dtype)),
                x_1=self.potential.dual(x_1.to(self.compute_dtype)),
                t=t.to(self.compute_dtype),
            )
            x_t = self.potential.legendre_map(dual_sample.x_t, **self.solver_kwargs)
            dx_t = self.potential.velocity(x_t, dual_sample.dx_t)
        return PathSample(x_1=x_1, x_0=x_0, t=t, x_t=x_t.to(dtype), dx_t=dx_t.to(dtype))
