"""Backing store for influence matrix: memmap or in-memory, read-only slices."""

import pathlib
from typing import Union

import numpy as np


class MemmapBackingStore:
    """Read-only backing store for the influence matrix. Uses memmap so the full matrix is not loaded into RAM."""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        shape: tuple,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.path = pathlib.Path(path)
        self._shape = shape
        self._dtype = dtype
        self._mmap: np.memmap = np.memmap(
            str(self.path),
            dtype=dtype,
            mode="r",
            shape=shape,
        )

    @property
    def shape(self) -> tuple:
        return self._shape

    def read_slice(
        self,
        r_lo: int,
        r_hi: int,
        d_lo: int,
        d_hi: int,
    ) -> np.ndarray:
        """Return a copy of the slice [r_lo:r_hi, d_lo:d_hi] (so caller cannot alter the backing array)."""
        return np.array(
            self._mmap[r_lo:r_hi, d_lo:d_hi],
            dtype=np.float32,
            copy=True,
        )

    def read_cell(self, r: int, d: int) -> float:
        return float(self._mmap[r, d])


class InMemoryBackingStore:
    """Backing store wrapping an in-memory numpy array (e.g. for tests)."""

    def __init__(self, array: np.ndarray) -> None:
        self._arr = np.asarray(array, dtype=np.float32)
        self._shape = self._arr.shape

    @property
    def shape(self) -> tuple:
        return self._shape

    def read_slice(
        self,
        r_lo: int,
        r_hi: int,
        d_lo: int,
        d_hi: int,
    ) -> np.ndarray:
        # np.asarray(..., copy=...) requires NumPy 2+; np.array supports copy= on 1.x.
        return np.array(
            self._arr[r_lo:r_hi, d_lo:d_hi],
            dtype=np.float32,
            copy=True,
        )

    def read_cell(self, r: int, d: int) -> float:
        return float(self._arr[r, d])
