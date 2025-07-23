import unittest

import numpy as np

from src.helpers.image_chunker_tree import build_chunking_tree, reconstruct_image


class TestImageChunker(unittest.TestCase):
    def test_image_chunker(self) -> None:
        image = np.random.rand(300, 100, 91)
        target_chunks = 8
        self.assertEqual(
            reconstruct_image(build_chunking_tree(image, target_chunks), target_chunks)
            + 1,
            image + 1,
            "Reconstructed image and the original image should be same",
        )
