from enum import Enum


class FilterType(Enum):
    UNIFORM = "uniform filter"
    GAUSSIAN = "normal gaussian filter"
    MEDIAN = "median filter"
    CUSTOM = "custom filter"
