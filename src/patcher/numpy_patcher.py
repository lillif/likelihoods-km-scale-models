import numpy as np


class NumpyPatcherMultipleArrays:
    def __init__(self, arrays: list[np.ndarray], patch_size: int, stride: int):
        for array in arrays:
            if array.shape != arrays[0].shape:
                raise ValueError("All arrays must have the same shape.")
        self.arrays = arrays
        self.patch_size = patch_size
        self.stride = stride

    def get_patches(self):
        for i in range(0, self.arrays[0].shape[0] - self.patch_size + 1, self.stride):
            for j in range(
                0, self.arrays[0].shape[1] - self.patch_size + 1, self.stride
            ):
                yield [
                    arr[i : i + self.patch_size, j : j + self.patch_size]
                    for arr in self.arrays
                ]
