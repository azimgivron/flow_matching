# Copyright (c) 2026 azim-givron
#
# This source code is licensed under the MIT license.
# SPDX-License-Identifier: MIT

"""Numerical solvers for the simplex-constrained Legendre map.

Ported from hessian-flow (``paths/geometries/lagrange_solvers.py``).
"""

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor

# A step function maps (x, direction, alpha) to the trial iterate, where alpha
# has shape x.shape[:-1].
StepFunction = Callable[[Tensor, Tensor, Tensor], Tensor]


def additive_step(x: Tensor, direction: Tensor, alpha: Tensor) -> Tensor:
    r"""Take the primal step :math:`x + \alpha d`.

    Args:
        x (Tensor): current iterate, shape (..., dim).
        direction (Tensor): Newton direction, same shape as ``x``.
        alpha (Tensor): step sizes, shape ``x.shape[:-1]``.

    Returns:
        Tensor: trial iterate, same shape as ``x``.
    """
    return x + alpha.unsqueeze(-1) * direction


def multiplicative_step(x: Tensor, direction: Tensor, alpha: Tensor) -> Tensor:
    r"""Take the step in log coordinates, :math:`\mathrm{softmax}(\log x + \alpha d \oslash x)`.

    For a tangent direction :math:`d` (summing to zero), :math:`d \oslash x` is the
    Newton direction in :math:`z = \log x`. The trial iterate has the same first-order
    change :math:`\alpha d` as the additive step, and it stays in the open simplex for
    every :math:`\alpha`.

    Args:
        x (Tensor): current positive iterate, shape (..., dim).
        direction (Tensor): tangent Newton direction, same shape as ``x``.
        alpha (Tensor): step sizes, shape ``x.shape[:-1]``.

    Returns:
        Tensor: trial iterate in the open simplex, same shape as ``x``.
    """
    return torch.softmax(x.log() + alpha.unsqueeze(-1) * direction / x, dim=-1)


@dataclass
class NewtonStats:
    r"""Work done by :func:`newton_lagrange`, summed over the batch.

    Attributes:
        iterations (int): Newton iterations (batched direction solves).
        objective_evaluations (int): batched objective evaluations in the line searches.
    """

    iterations: int = 0
    objective_evaluations: int = 0


def _quadratic_interpolation_step(
    alpha: Tensor,
    f0: Tensor,
    dir_deriv: Tensor,
    candidate_value: Tensor,
    beta: float,
    min_step: float,
) -> Tensor:
    r"""Compute a safeguarded quadratic interpolation step.

    Invalid estimates fall back to the contracted step :math:`\beta\alpha`.

    Args:
        alpha (Tensor): current trial step sizes, shape (...).
        f0 (Tensor): objective values at the base iterate, shape (...).
        dir_deriv (Tensor): directional derivatives at the base iterate, shape (...).
        candidate_value (Tensor): objective values at the trial points, shape (...).
        beta (float): backtracking contraction factor.
        min_step (float): lower bound on admissible step sizes.

    Returns:
        Tensor: next trial step sizes, shape (...).
    """
    denom = 2.0 * (candidate_value - f0 - alpha * dir_deriv)
    quadratic_alpha = -dir_deriv * alpha.square() / denom
    fallback_alpha = beta * alpha
    safeguarded_alpha = torch.where(
        torch.isfinite(quadratic_alpha) & (quadratic_alpha > min_step),
        quadratic_alpha,
        fallback_alpha,
    )
    return torch.minimum(torch.clamp(safeguarded_alpha, min=min_step), 0.5 * alpha)


def _cubic_interpolation_step(
    previous_alpha: Tensor,
    alpha: Tensor,
    f0: Tensor,
    previous_value: Tensor,
    candidate_value: Tensor,
    dir_deriv: Tensor,
    beta: float,
    min_step: float,
) -> Tensor:
    r"""Compute a safeguarded cubic interpolation step.

    The cubic model is fitted through the base point and the two most recent rejected
    trial steps. Invalid estimates fall back to the contracted step :math:`\beta\alpha`.

    Args:
        previous_alpha (Tensor): previous rejected trial step sizes, shape (...).
        alpha (Tensor): current trial step sizes, shape (...).
        f0 (Tensor): objective values at the base iterate, shape (...).
        previous_value (Tensor): objective values at the previous trial points, shape (...).
        candidate_value (Tensor): objective values at the current trial points, shape (...).
        dir_deriv (Tensor): directional derivatives at the base iterate, shape (...).
        beta (float): backtracking contraction factor.
        min_step (float): lower bound on admissible step sizes.

    Returns:
        Tensor: next trial step sizes, shape (...).
    """
    d1 = previous_value - f0 - previous_alpha * dir_deriv
    d2 = candidate_value - f0 - alpha * dir_deriv
    denom = (previous_alpha - alpha) * previous_alpha.square() * alpha.square()

    coeff_a = (previous_alpha.square() * d2 - alpha.square() * d1) / denom
    coeff_b = (-previous_alpha.pow(3) * d2 + alpha.pow(3) * d1) / denom
    discriminant = coeff_b.square() - 3.0 * coeff_a * dir_deriv

    cubic_alpha = (-coeff_b + torch.sqrt(torch.clamp(discriminant, min=0.0))) / (
        3.0 * coeff_a
    )
    fallback_alpha = beta * alpha
    valid_cubic_mask = (
        torch.isfinite(cubic_alpha)
        & torch.isfinite(coeff_a)
        & (coeff_a != 0)
        & (cubic_alpha > min_step)
    )
    safeguarded_alpha = torch.where(valid_cubic_mask, cubic_alpha, fallback_alpha)
    return torch.minimum(torch.clamp(safeguarded_alpha, min=min_step), 0.5 * alpha)


def armijo_backtracking(
    func: Callable[[Tensor], Tensor],
    x0: Tensor,
    dir_deriv: Tensor,
    dk: Tensor,
    sigma: float,
    beta: float,
    max_iters: int = 50,
    min_step: float = 1e-12,
    stationary_mask: Optional[Tensor] = None,
    step: StepFunction = additive_step,
) -> Tensor:
    r"""Run batched Armijo backtracking with safeguarded interpolation.

    A trial point is accepted when its objective is no larger than
    :math:`f_0 + \sigma\alpha\,\partial f`. Rejected steps are reduced by quadratic
    (first rejection) or cubic (later rejections) interpolation.

    Adapted from https://github.com/hjmshi/PyTorch-LBFGS/blob/master/functions/LBFGS.py.

    Args:
        func (Callable[[Tensor], Tensor]): batched objective, output shape (...).
        x0 (Tensor): current iterate, shape (..., dim).
        dir_deriv (Tensor): directional derivative at ``x0``, shape ``x0.shape[:-1]``.
        dk (Tensor): search direction, same shape as ``x0``.
        sigma (float): Armijo parameter in (0, 1).
        beta (float): backtracking contraction factor in (0, 1).
        max_iters (int): maximum number of backtracking steps.
        min_step (float): minimum admissible step size.
        stationary_mask (Optional[Tensor]): boolean mask, shape ``x0.shape[:-1]``. Marked members are kept at ``x0``.
        step (StepFunction): maps ``(x0, dk, alpha)`` to the trial iterate; ``dir_deriv`` must be its derivative in ``alpha`` at zero.

    Raises:
        ValueError: if parameters are out of range or an active direction is not a descent direction.
        RuntimeError: if the objective is non-finite, a trial update stagnates, the minimum step is reached, or ``max_iters`` is exhausted.

    Returns:
        Tensor: accepted iterate, same shape as ``x0``.
    """
    if not 0 < sigma < 1:
        raise ValueError(f"sigma must be in (0, 1), but equals {sigma}")
    if not 0 < beta < 1:
        raise ValueError(f"beta must be in (0, 1), but equals {beta}")

    if stationary_mask is None:
        stationary_mask = torch.zeros_like(dir_deriv, dtype=torch.bool)
    if stationary_mask.shape != dir_deriv.shape:
        raise ValueError("stationary_mask must have the same shape as dir_deriv.")
    if torch.any((dir_deriv >= 0) & ~stationary_mask):
        raise ValueError("Armijo line search requires a descent direction.")

    alpha = torch.ones_like(dir_deriv)
    f0 = func(x0)
    if not torch.all(torch.isfinite(f0)):
        raise RuntimeError(
            "Line search received a non-finite objective at the current iterate."
        )

    previous_alpha = torch.zeros_like(alpha)
    previous_value = f0.clone()

    for _ in range(max_iters):
        candidate = torch.where(stationary_mask.unsqueeze(-1), x0, step(x0, dk, alpha))
        if torch.any(~stationary_mask & torch.all(candidate == x0, dim=-1)):
            raise RuntimeError(
                "Armijo line search stagnated: a trial update no longer changes "
                "the primal iterate in the current dtype."
            )
        candidate_value = func(candidate)
        legal_mask = torch.isfinite(candidate_value)
        sufficient_decrease_mask = stationary_mask | (
            legal_mask & (candidate_value <= f0 + sigma * alpha * dir_deriv)
        )
        if torch.all(sufficient_decrease_mask):
            return candidate

        rejected_mask = ~sufficient_decrease_mask
        first_backtrack_mask = rejected_mask & (previous_alpha == 0)
        later_backtrack_mask = rejected_mask & (previous_alpha > 0)
        new_alpha = alpha.clone()

        if torch.any(first_backtrack_mask):
            new_alpha[first_backtrack_mask] = _quadratic_interpolation_step(
                alpha[first_backtrack_mask],
                f0[first_backtrack_mask],
                dir_deriv[first_backtrack_mask],
                candidate_value[first_backtrack_mask],
                beta,
                min_step,
            )

        if torch.any(later_backtrack_mask):
            new_alpha[later_backtrack_mask] = _cubic_interpolation_step(
                previous_alpha[later_backtrack_mask],
                alpha[later_backtrack_mask],
                f0[later_backtrack_mask],
                previous_value[later_backtrack_mask],
                candidate_value[later_backtrack_mask],
                dir_deriv[later_backtrack_mask],
                beta,
                min_step,
            )

        previous_alpha = torch.where(rejected_mask, alpha, previous_alpha)
        previous_value = torch.where(rejected_mask, candidate_value, previous_value)
        alpha = torch.where(rejected_mask, new_alpha, alpha)
        next_candidate = step(x0, dk, alpha)
        if torch.any(
            rejected_mask & ~stationary_mask & torch.all(next_candidate == x0, dim=-1)
        ):
            raise RuntimeError(
                "Armijo line search stagnated: a trial update no longer changes "
                "the primal iterate in the current dtype."
            )
        if torch.any(rejected_mask & (alpha <= min_step)):
            raise RuntimeError(
                "Line search failed: step size reached the minimum threshold."
            )
    raise RuntimeError("Line search failed: maximum backtracking iterations reached.")


def schur_complement(
    B: Tensor, A: Tensor, g: Tensor, b: Tensor, *, check_errors: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Solve the equality-constrained system :math:`Bx + A\lambda = -g`, :math:`A^\top x = b`.

    :math:`B` is assumed positive definite. :math:`x` is eliminated first, the
    Schur-complement system is solved for :math:`\lambda`, then :math:`x` is recovered
    by a second solve against :math:`B` (more accurate near convergence than reusing
    the stored solves).

    Args:
        B (Tensor): positive-definite block, shape (..., n, n).
        A (Tensor): constraint block, shape (..., n, m).
        g (Tensor): primal right-hand side, shape (..., n).
        b (Tensor): constraint right-hand side, shape (..., m).
        check_errors (bool): raise if a Cholesky factorization fails.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: ``x`` of shape (..., n), ``lambda`` of shape (..., m), and a boolean mask of shape (...) of the systems whose factorizations succeeded.
    """
    cholesky_B, info_B = torch.linalg.cholesky_ex(B, check_errors=check_errors)
    B_inv_A, B_inv_g = torch.cholesky_solve(
        torch.cat((A, g.unsqueeze(-1)), dim=-1), cholesky_B
    ).split((A.size(-1), 1), dim=-1)

    schur_matrix = A.transpose(-1, -2) @ B_inv_A
    schur_rhs = -(A.transpose(-1, -2) @ B_inv_g + b.unsqueeze(-1))
    cholesky_schur, info_schur = torch.linalg.cholesky_ex(
        schur_matrix, check_errors=check_errors
    )
    lmbd = torch.cholesky_solve(schur_rhs, cholesky_schur)

    x = torch.cholesky_solve(-(g.unsqueeze(-1) + A @ lmbd), cholesky_B)
    return x.squeeze(-1), lmbd.squeeze(-1), (info_B == 0) & (info_schur == 0)


def tangent_solve(hessian: Tensor, rhs: Tensor) -> Tensor:
    r"""Solve :math:`G w + \eta\mathbf{1} = r` with :math:`\mathbf{1}^\top w = 0`.

    The solution is :math:`w = H r` where
    :math:`H = G^{-1} - G^{-1}\mathbf{1}\mathbf{1}^\top G^{-1} / \mathbf{1}^\top G^{-1}\mathbf{1}`
    is the inverse of the metric on the tangent space (paper, Lemma 1).

    Args:
        hessian (Tensor): positive-definite matrices :math:`G`, shape (..., dim, dim).
        rhs (Tensor): right-hand sides :math:`r`, shape (..., dim).

    Returns:
        Tensor: tangent solutions :math:`w`, shape (..., dim).
    """
    ones = torch.ones(*rhs.shape, 1, device=rhs.device, dtype=rhs.dtype)
    zeros = torch.zeros(*rhs.shape[:-1], 1, device=rhs.device, dtype=rhs.dtype)
    w, _, _ = schur_complement(hessian, ones, -rhs, zeros, check_errors=True)
    return w


def newton_lagrange(
    x0: Tensor,
    dual: Tensor,
    *,
    objective: Callable[[Tensor, Tensor], Tensor],
    gradient: Callable[[Tensor], Tensor],
    hessian: Callable[[Tensor], Tensor],
    step: StepFunction = additive_step,
    tol: float = 1e-4,
    max_iters: int = 100,
    sigma: float = 1e-4,
    beta: float = 0.5,
    armijo_kwargs: Optional[Mapping[str, Union[float, int]]] = None,
    return_stats: bool = False,
) -> Union[Tensor, Tuple[Tensor, NewtonStats]]:
    r"""Solve the simplex-constrained Legendre problem by damped Newton steps.

    Finds :math:`x` such that :math:`\nabla\phi(x) - \gamma + \eta\mathbf{1} = 0` and
    :math:`\mathbf{1}^\top x = 1` (paper, Eq. 5). Each iteration solves the
    Newton–Lagrange system (paper, Eq. 27) and applies Armijo backtracking on
    :math:`\phi(x) - \langle\gamma, x\rangle` along ``step``. Every simplex is solved
    independently; converged ones are removed from later iterations.

    Args:
        x0 (Tensor): initial iterate, shape (..., dim).
        dual (Tensor): target dual coordinates :math:`\gamma`, same shape as ``x0``.
        objective (Callable[[Tensor, Tensor], Tensor]): maps ``(x, dual)`` of shape (n, dim) to :math:`\phi(x) - \langle\gamma, x\rangle`, shape (n,).
        gradient (Callable[[Tensor], Tensor]): :math:`\nabla\phi`, maps (n, dim) to (n, dim).
        hessian (Callable[[Tensor], Tensor]): :math:`\nabla^2\phi`, maps (n, dim) to (n, dim, dim).
        step (StepFunction): how a Newton direction is applied: :func:`additive_step` or :func:`multiplicative_step` (log coordinates; ``x0`` must be positive).
        tol (float): infinity-norm tolerance on the Lagrange residual, floored at the dtype's machine epsilon. In float64, tolerances below about ``1e-7`` can make the Armijo test fail: the objective decrease then falls below the floating-point resolution.
        max_iters (int): maximum Newton iterations.
        sigma (float): Armijo parameter.
        beta (float): backtracking contraction factor.
        armijo_kwargs (Optional[Mapping[str, Union[float, int]]]): overrides for Armijo's ``sigma``, ``beta``, ``max_iters`` and ``min_step``.
        return_stats (bool): also return the work done as :class:`NewtonStats`.

    Raises:
        ValueError: if shapes, tolerances or ``armijo_kwargs`` are invalid.
        RuntimeError: if the iterations do not converge.

    Returns:
        Union[Tensor, Tuple[Tensor, NewtonStats]]: the solution, same shape as ``x0``, and the work done when ``return_stats`` is set.
    """
    if dual.shape != x0.shape:
        raise ValueError("dual must have the same shape as x0.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    tol = max(torch.finfo(x0.dtype).eps, tol)

    armijo_options: Dict[str, Union[float, int]] = {"sigma": sigma, "beta": beta}
    if armijo_kwargs is not None:
        invalid_keys = set(armijo_kwargs) - {"sigma", "beta", "max_iters", "min_step"}
        if invalid_keys:
            raise ValueError(
                "armijo_kwargs only supports sigma, beta, max_iters, and min_step."
            )
        armijo_options.update(armijo_kwargs)

    shape = x0.shape
    x = x0.reshape(-1, shape[-1]).clone()
    dual = dual.reshape(-1, shape[-1])
    active_mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    stats = NewtonStats()

    def counted_objective(trial_x: Tensor, current_dual: Tensor) -> Tensor:
        stats.objective_evaluations += 1
        return objective(trial_x, current_dual)

    for _ in range(max_iters):
        x_active = x[active_mask]
        dual_active = dual[active_mask]
        gradient_active = gradient(x_active)
        lmbd_active = (dual_active - gradient_active).mean(dim=-1, keepdim=True)

        residual = torch.cat(
            [
                gradient_active - dual_active + lmbd_active,
                (x_active.sum(dim=-1, keepdim=True) - 1),
            ],
            dim=-1,
        )
        converged = torch.linalg.vector_norm(residual, ord=float("inf"), dim=-1) < tol
        active_index = active_mask.nonzero().squeeze(-1)
        active_mask[active_index[converged]] = False

        if not torch.any(active_mask):
            x = x.reshape(shape)
            return (x, stats) if return_stats else x
        stats.iterations += 1

        x_active = x_active[~converged]
        dual_active = dual_active[~converged]
        gradient_active = gradient_active[~converged]

        ones = torch.ones(*x_active.shape, 1, device=x.device, dtype=x.dtype)
        direction, _, _ = schur_complement(
            hessian(x_active),
            ones,
            gradient_active - dual_active,
            1 - x_active.sum(dim=-1, keepdim=True),
            check_errors=True,
        )
        dir_deriv = ((gradient_active - dual_active) * direction).sum(dim=-1)
        x[active_mask] = armijo_backtracking(
            lambda trial_x, current_dual=dual_active: counted_objective(
                trial_x, current_dual
            ),
            x_active,
            dir_deriv,
            direction,
            step=step,
            **armijo_options,
        )
    raise RuntimeError(
        "Newton-Lagrange algorithm did not converge within the maximum number of iterations."
    )
