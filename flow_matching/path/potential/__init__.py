# Copyright (c) 2026 azim-givron
#
# This source code is licensed under the MIT license.
# SPDX-License-Identifier: MIT

from .kernel_smoothed_entropy import KernelSmoothedEntropy
from .negative_entropy import NegativeEntropy
from .potential import BregmanPotential, LogBarrierPotential
from .solvers import (
    additive_step,
    armijo_backtracking,
    multiplicative_step,
    newton_lagrange,
    NewtonStats,
    schur_complement,
    tangent_solve,
)

__all__ = [
    "BregmanPotential",
    "LogBarrierPotential",
    "NegativeEntropy",
    "KernelSmoothedEntropy",
    "additive_step",
    "multiplicative_step",
    "armijo_backtracking",
    "newton_lagrange",
    "NewtonStats",
    "schur_complement",
    "tangent_solve",
]
