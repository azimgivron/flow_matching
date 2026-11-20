``flow_matching.path``
=============================

.. currentmodule:: flow_matching.path


Probability Paths
--------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ProbPath
   AffineProbPath
   CondOTProbPath
   MixtureDiscreteProbPath
   GeodesicProbPath
   HessianProbPath


Label Smoothing
--------------------------------

Maps categorical data into the open simplex before training a :class:`HessianProbPath`
(see also :doc:`flow_matching.path.potential`).

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   LabelSmoothing
   DirichletSmoothing


Path Sample
--------------------------------

Corresponds to an instance of a sample drawn from the probability path.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   path_sample.PathSample
   path_sample.DiscretePathSample
   
