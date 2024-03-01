from pydantic import BaseModel, NonnegtiveInt
from enum import Enum
import numpy as np
class FilterType(Enum):
    UNIFORM =  "uniform filter",
    GAUSSIAN =  "normal gaussian filter",
    MEDIAN =  "median filter",
    CUSTOM =  "custom filter",

class ImageFilters(BaseModel):
    image: np.ndarray
    window_size: NonnegtiveInt
    filter_type: FilterType
