from enum import Enum

import numpy as np
from pydantic import BaseModel, NonNegativeInt


class FilterType(Enum):
    UNIFORM = "uniform filter"
    GAUSSIAN = "normal gaussian filter"
    MEDIAN = "median filter"
    CUSTOM = "custom filter"
