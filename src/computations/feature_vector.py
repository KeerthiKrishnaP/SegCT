import numpy as np

from helpers.parallel_process_big_images import Prallel_Process_Image
from models.outils import Operations


class FeatureVector:
    def __init__(self, image: np.ndarray, window_radius: int) -> None:
        self.image = image
        self.window_radius = window_radius
        self.image_process = Prallel_Process_Image(
            image=image,
            chunk_shape=(2, 2, 2),
            number_of_processes=4,
            window_size=window_radius,
            operation_name=None,
        )
        self.intensity = self.compute_intensity().result
        self.orientation = self.compute_orientations().result
        self.anisotropy = self.compute_anisotropy().result

    def compute_intensity(self) -> Prallel_Process_Image:
        self.image_process.operation_name = Operations.CST
        return self.image_process

    def compute_orientations(self) -> Prallel_Process_Image:
        self.image_process.operation_name = Operations.CEV
        return self.image_process

    def compute_anisotropy(self) -> Prallel_Process_Image:
        self.image_process.operation_name = Operations.CFV
        return self.image_process
