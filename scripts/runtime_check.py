from __future__ import annotations

import json
import os
import sys

# Import the Agent module first: on Windows it registers the Pixi/Conda
# Library/bin directory before NumPy loads its BLAS runtime.
from starter.agent import DEFAULT_MODEL_NAME, _WINDOWS_DLL_DIRECTORY

import numpy as np


def main() -> None:
    left = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    right = np.asarray([1.0, -1.0], dtype=np.float32)
    dot_result = (left @ right).tolist()

    torch_summary: dict[str, object]
    try:
        import torch

        torch_summary = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }
    except ImportError:
        torch_summary = {"available": False}

    print(
        json.dumps(
            {
                "python": sys.executable,
                "prefix": sys.prefix,
                "conda_prefix": os.environ.get("CONDA_PREFIX"),
                "default_model": DEFAULT_MODEL_NAME,
                "numpy": np.__version__,
                "blas_dot_result": dot_result,
                "windows_dll_directory_registered": (
                    _WINDOWS_DLL_DIRECTORY is not None
                    if os.name == "nt"
                    else None
                ),
                "torch": torch_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
