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

# Re-export for backward compatibility
__all__ += ["__version__"]


# Add classmethods to the PyO3 Model class for backward compatibility with the old API
Model.default_hyper_param = classmethod(lambda cls: default_hyper_param())
Model.train_on_data = classmethod(
    lambda cls, data_path, hyper_param=None, n_threads=None: train_on_data(
        data_path, hyper_param, n_threads
    )
)
