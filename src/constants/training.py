import numpy as np

# default split configuration for the datamodule
SPLITS_DICT = {
    "train": {
        "years": [2024],
        "months": np.arange(1, 13).tolist(),
        "days": np.arange(1, 16).tolist(),
    },
    "val": {
        "years": [2024],
        "months": np.arange(1, 13).tolist(),
        "days": np.arange(20, 23).tolist(),
    },
    "test": {
        "years": [2024],
        "months": np.arange(1, 13).tolist(),
        "days": np.arange(25, 28).tolist(),
    },
}

# bin width used for histogram matching
HISTOGRAM_RESOLUTION = 0.05  # W/m²
