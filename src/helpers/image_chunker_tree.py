import numpy as np


class TreeNode:
    def __init__(self, chunk, depth=0, name=None, start_indices=None):
        self.chunk = chunk
        self.children = []
        self.depth = depth
        self.name = name or f"Chunk_{depth}_{id(self)}"
        self.start_indices = start_indices or [0] * len(chunk.shape)

    def is_leaf(self):
        return len(self.children) == 0


def chunk_image(node, target_chunks, current_chunks):
    if current_chunks >= target_chunks:
        return current_chunks

    chunk = node.chunk
    shape = chunk.shape
    max_dim = shape.index(max(shape))
    split_index = shape[max_dim] // 2

    if max_dim == 0:
        chunk1 = chunk[:split_index, :, :]
        chunk2 = chunk[split_index:, :, :]
    elif max_dim == 1:
        chunk1 = chunk[:, :split_index, :]
        chunk2 = chunk[:, split_index:, :]
    else:
        chunk1 = chunk[:, :, :split_index]
        chunk2 = chunk[:, :, split_index:]

    start_indices1 = node.start_indices.copy()
    start_indices2 = node.start_indices.copy()
    start_indices2[max_dim] += split_index

    child1 = TreeNode(chunk1, node.depth + 1, start_indices=start_indices1)
    child2 = TreeNode(chunk2, node.depth + 1, start_indices=start_indices2)
    node.children = [child1, child2]

    current_chunks += 2

    current_chunks = chunk_image(child1, target_chunks, current_chunks)
    current_chunks = chunk_image(child2, target_chunks, current_chunks)

    return current_chunks


def build_chunking_tree(image, target_chunks):
    root = TreeNode(image)
    chunk_image(root, target_chunks, 1)
    return root


def reconstruct_image(root, original_shape):
    image = np.zeros(original_shape)

    def place_chunk(chunk, start_indices):
        slices = tuple(
            slice(start, start + s) for start, s in zip(start_indices, chunk.shape)
        )
        image[slices] = chunk

    def add_chunks(node):
        place_chunk(node.chunk, node.start_indices)
        for child in node.children:
            add_chunks(child)

    add_chunks(root)

    return image
