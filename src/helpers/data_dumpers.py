import json
import os
from pathlib import Path
from typing import Final

import numpy as np


def clear_folder(path: Path) -> None:
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


STRUCTURE_TENSOR_DIRECTORY = "src/saved_model_varibles/image_data"


def create_folder_if_not_exists(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):  # -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
