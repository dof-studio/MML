# __init__.py
#
# Importing functionality
# From MML Library by Nathmath

# Macro------------------------------------------------------------------------
__version__ = "0.1.1"
__all__ = ['Tensor', "Matrix", "GradientOptimizer", "optimize"]

# Production ------------------------------------------------------------------
mml_production = True

# Import-----------------------------------------------------------------------


# Tensor class
if mml_production:
    import mml.tensor
    from mml.tensor import Tensor
else:
    import tensor
    from tensor import Tensor

# Matrix class
if mml_production:
    import mml.matrix
    from mml.matrix import Matrix
else:
    import matrix
    from matrix import Matrix

# Optimizer class
if mml_production:
    import mml.optimizer
    from mml.optimizer import GradientOptimizer, optimize
else:
    import optimizer
    from optimizer import GradientOptimizer, optimize

