from enum import Enum
from typing import List

from pydantic import BaseModel, ValidationError, validator


class EvenNumberListInput(BaseModel):
    numbers: List[int]

    @validator("numbers")
    def check_even_numbers(cls, values) -> list[int]:
        if any(number % 2 != 0 for number in values):
            raise ValueError("List must contain only even numbers")
        return values


class Operations(Enum):
    CST = "COMPUTE STRUCTURE TENSOR"
    CEV = "COMPUTE EIGEN VALUES"
    CFV = "COMPUTE FEATURE VECTORS"
