from pathlib import Path
import re
from typing import Optional
import numpy as np
import numpy.linalg as nlinalg
import scipy.linalg as slinalg

ROOT_DIR = Path(__file__).resolve()


def _diagonal_indices(size: int, k: int = 0):
    rows, cols = np.diag_indices(size)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


class Operator:
    def __init__(self, I: float, nspins: int = 1, matrix: Optional[np.ndarray] = None):
        self.dim = self._get_dim(I, nspins=nspins)
        self.I = I
        self.nspins = nspins
        if matrix is None:
            self.matrix = np.zeros((self.dim, self.dim))
        else:
            assert all([self.dim == matrix.shape[i] for i in range(2)])
            self.matrix = matrix

    def __str__(self):
        return str(self.matrix)

    def __eq__(self, other):
        return (self.I == other.I) and np.array_equal(self.matrix, other.matrix)

    def __add__(self, other):
        matrix = self.matrix + other.matrix
        return Operator(self.I, nspins=self.nspins, matrix=matrix)

    def __sub__(self, other):
        matrix = self.matrix - other.matrix
        return Operator(self.I, nspins=self.nspins, matrix=matrix)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            matrix = self.matrix * other
        elif isinstance(other, Operator):
            matrix = self.matrix * self.other
        return Operator(self.I, nspins=self.nspins, matrix=matrix)

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            matrix = other * self.matrix
        elif isinstance(other, Operator):
            matrix = self.other * self.matrix
        return Operator(self.I, nspins=self.nspins, matrix=matrix)

    def __matmul__(self, other):
        # Extend for state vectors too
        matrix = self.matrix @ other.matrix
        return Operator(self.I, nspins=self.nspins, matrix=matrix)

    def __pow__(self, power):
        matrix = nlinalg.matrix_power(self.matrix, power)
        return Operator(self.I, nspins=self.nspins, matrix=matrix)

    def commutator(self, other):
        """Compute [self, other]."""
        self._assert_same_spin_and_shape(other)
        return (self @ other) - (other @ self)

    def kroenecker(self, other):
        self._assert_same_spin(other)
        matrix = np.kron(self.matrix, other.matrix)
        nspins = self.nspins + other.nspins
        return Operator(self.I, nspins=nspins, matrix=matrix)

    def rotation_operator(self, angle: float):
        return Operator(
            self.I, nspins=self.nspins, matrix=slinalg.expm(-1j * angle * self.matrix)
        )

    def _assert_same_spin(self, other):
        assert self.I == other.I

    def _assert_same_spin_and_shape(self, other):
        assert (self.I == other.I) and (self.nspins == other.nspins)

    @classmethod
    def Ix(cls, I: float):
        dim = cls._get_dim(I)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[_diagonal_indices(dim, k=1)] = 0.5 * np.sqrt(
            [I * (I + 1) - m * (m + 1) for m in np.linspace(-I, I - 1, dim - 1)]
        )
        matrix[_diagonal_indices(dim, k=-1)] = 0.5 * np.sqrt(
            [I * (I + 1) - m * (m - 1) for m in np.linspace(-I + 1, I, dim - 1)]
        )
        return cls(I, matrix=matrix)

    @classmethod
    def Iy(cls, I: float):
        dim = cls._get_dim(I)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[_diagonal_indices(dim, k=1)] = -0.5j * np.sqrt(
            [I * (I + 1) - m * (m + 1) for m in np.linspace(-I, I - 1, dim - 1)]
        )
        matrix[_diagonal_indices(dim, k=-1)] = 0.5j * np.sqrt(
            [I * (I + 1) - m * (m - 1) for m in np.linspace(-I + 1, I, dim - 1)]
        )
        return cls(I, matrix=matrix)

    @classmethod
    def Iz(cls, I: float):
        dim = cls._get_dim(I)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[_diagonal_indices(dim)] = np.round(np.linspace(I, -I, dim), 1)
        return cls(I, matrix=matrix)

    @classmethod
    def E(cls, I: float):
        dim = cls._get_dim(I)
        return cls(I, matrix=0.5 * np.eye(dim, dtype="complex"))

    @classmethod
    def Iplus(cls, I: float):
        dim = cls._get_dim(I)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[_diagonal_indices(dim, k=1)] = np.sqrt(
            [I * (I + 1) - m * (m + 1) for m in np.linspace(-I, I - 1, dim - 1)]
        )
        return cls(I, matrix=matrix)

    @classmethod
    def Iminus(cls, I: float):
        dim = cls._get_dim(I)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[_diagonal_indices(dim, k=-1)] = np.sqrt(
            [I * (I + 1) - m * (m - 1) for m in np.linspace(-I + 1, I, dim - 1)]
        )
        return cls(I, matrix=matrix)

    @staticmethod
    def _get_dim(I: float, nspins: int = 1):
        assert I % 0.5 == 0.0
        return int(2 * I + 1) * nspins


class CartesianBasis:
    def __init__(self, I: float, nspins: int = 1):
        assert I % 0.5 == 0.0
        assert nspins >= 1
        self._basis_operators = {
            "x": Operator.Ix(I),
            "y": Operator.Iy(I),
            "z": Operator.Iz(I),
            "e": Operator.E(I),
        }
        self.nspins = nspins

    def get(self, operator: str) -> np.ndarray:
        assert re.match(r"(\d+(x|y|z))+$", operator)
        elements = {}
        for component in re.findall(r"\d+(?:x|y|z)", operator):
            num = int(re.search(r"\d+", component).group(0))
            coord = re.search(r"(x|y|z)", component).group(0)
            assert num <= self.nspins
            assert num not in elements
            elements[num] = coord

        for i in range(1, self.nspins + 1):
            if i == 1:
                if i in elements:
                    coord = elements[i]
                    matrix = self._basis_operators[coord]
                else:
                    matrix = self._basis_operators["e"]
            else:
                if i in elements:
                    coord = elements[i]
                    matrix = matrix.kroenecker(self._basis_operators[coord])
                else:
                    matrix = matrix.kroenecker(self._basis_operators["e"])

        return matrix
