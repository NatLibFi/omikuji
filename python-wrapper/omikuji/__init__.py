__version__ = "0.5.1"
__all__ = [
    "Model",
    "LossType",
    "HyperParam",
    "default_hyper_param",
    "train_on_data",
    "init_logger",
]

# Import from PyO3 native module
from omikuji._omikuji import (
    Model,
    LossType,
    HyperParam,
    default_hyper_param,
    train_on_data,
    init_logger,
)

# Backward-compatible aliases for the old LossType member names
LossType.HINGE = LossType.Hinge
LossType.LOG = LossType.Log

# Re-export for backward compatibility
__all__ += ["__version__"]
