``flow_matching.path.potential``
=================================

.. currentmodule:: flow_matching.path.potential

Convex potentials (Bregman generators) defining the geometry of :class:`~flow_matching.path.HessianProbPath`.
New potentials subclass :class:`BregmanPotential` (or :class:`LogBarrierPotential` for potentials with a
logarithmic barrier) and implement ``forward``, ``gradient`` and ``hessian``.


Potentials
--------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BregmanPotential
   LogBarrierPotential
   NegativeEntropy
   KernelSmoothedEntropy


Legendre map solvers
--------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   newton_lagrange
   armijo_backtracking
   schur_complement
   tangent_solve
   additive_step
   multiplicative_step
   NewtonStats
