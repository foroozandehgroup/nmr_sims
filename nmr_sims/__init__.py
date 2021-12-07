from pathlib import Path
import re
from typing import Iterable, Optional
import numpy as np
import numpy.linalg as nlinalg
import scipy.linalg as slinalg

ROOT_DIR = Path(__file__).resolve()

class EigenState:
    def __init__(self, I: float, M: float):
        assert all([round(n % 0.5, 1) == 0.0 for n in (I, M)])
        assert M in range(-I, I + 1)
        self.I = I
        self.M = M
        self.factor = 1.0


class State:
    def __init__(self, I: float, coefficients: Iterable[float]):
        assert round(I % 0.5, 1) == 0.0
        assert len(coefficients) == int(2 * I + 1)
        csum = np.sqrt(np.sum([c ** 2 for c in coefficients]))
        coefficients = [c / csum for c in coefficients]
        self.state = {
            M : c for c, M in
            zip(coefficients, [m / 2 for m in range(-int(2 * I), int(2 * (I + 1)), 2)])
        }


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
            self.I, nspins=nspins, matrix=slinalg.expm(-1j * angle * self.matrix)
        )

    def _assert_same_spin(self, other):
        assert self.I == other.I

    def _assert_same_spin_and_shape(self, other):
        assert (self.I == other.I) and (self.nspins == other.nspins)

    @classmethod
    def Ix(cls, I: float):
        dim = cls._get_dim(I)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[_diagonal_indices(dim, k=1)] = \
            0.5 * np.sqrt(
                [I * (I + 1) - m * (m + 1) for m in np.linspace(-I, I - 1, dim - 1)]
            )
        matrix[_diagonal_indices(dim, k=-1)] = \
            0.5 * np.sqrt(
                [I * (I + 1) - m * (m - 1) for m in np.linspace(-I + 1, I, dim - 1)]
            )
        return cls(I, matrix=matrix)

    @classmethod
    def Iy(cls, I: float):
        dim = cls._get_dim(I)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[_diagonal_indices(dim, k=1)] = \
            -0.5j * np.sqrt(
                [I * (I + 1) - m * (m + 1) for m in np.linspace(-I, I - 1, dim - 1)]
            )
        matrix[_diagonal_indices(dim, k=-1)] = \
            0.5j * np.sqrt(
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
        matrix[_diagonal_indices(dim, k=1)] = \
            np.sqrt(
                [I * (I + 1) - m * (m + 1) for m in np.linspace(-I, I - 1, dim - 1)]
            )
        return cls(I, matrix=matrix)

    @classmethod
    def Iminus(cls, I: float):
        dim = cls._get_dim(I)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[_diagonal_indices(dim, k=-1)] = \
            np.sqrt(
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




class Test:
    def test_operator(self):
        sqrt2 = np.sqrt(2)
        sqrt3 = np.sqrt(3)
        sqrt5 = np.sqrt(5)
        Ix_half = Operator.Ix(0.5)
        assert np.array_equal(
            Ix_half.matrix, np.array([[0.0, 0.5], [0.5, 0.0]], dtype="complex")
        )
        Iy_half = Operator.Iy(0.5)
        assert np.array_equal(
            Iy_half.matrix, np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype="complex")
        )
        Iz_half = Operator.Iz(0.5)
        assert np.array_equal(
            Iz_half.matrix, np.array([[0.5, 0.0], [0.0, -0.5]], dtype="complex")
        )
        Iplus_half = Operator.Iplus(0.5)
        assert np.array_equal(
            Iplus_half.matrix, np.array([[0.0, 1.0], [0.0, 0.0]], dtype="complex")
        )
        Iminus_half = Operator.Iminus(0.5)
        assert np.array_equal(
            Iminus_half.matrix, np.array([[0.0, 0.0], [1.0, 0.0]], dtype="complex")
        )

        Ix_one = Operator.Ix(1)
        assert np.allclose(
            Ix_one.matrix,
            (1 / sqrt2) * np.array([
                [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]
            ], dtype="complex"),
            rtol=0.0,
            atol=1E-10,
        )
        Iy_one = Operator.Iy(1)
        assert np.allclose(
            Iy_one.matrix,
            (1j / sqrt2) * np.array([
                [0.0, -1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, 0.0]
            ], dtype="complex"),
            rtol=0.0,
            atol=1E-10,
        )
        Iz_one = Operator.Iz(1)
        assert np.allclose(
            Iz_one.matrix,
            np.array([
                [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]
            ], dtype="complex"),
            rtol=0.0,
            atol=1E-10,
        )

        Ix_three_halves = Operator.Ix(1.5)
        assert np.allclose(
            Ix_three_halves.matrix,
            0.5 * np.array([
                [0.0, sqrt3, 0.0, 0.0],
                [sqrt3, 0.0, 2.0, 0.0],
                [0.0, 2.0, 0.0, sqrt3],
                [0.0, 0.0, sqrt3, 0.0],
            ], dtype="complex"),
            rtol=0.0,
            atol=1E-10,
        )
        Iy_three_halves = Operator.Iy(1.5)
        assert np.allclose(
            Iy_three_halves.matrix,
            -0.5j * np.array([
                [0.0, sqrt3, 0.0, 0.0],
                [-sqrt3, 0.0, 2.0, 0.0],
                [0.0, -2.0, 0.0, sqrt3],
                [0.0, 0.0, -sqrt3, 0.0],
            ], dtype="complex"),
            rtol=0.0,
            atol=1E-10,
        )
        Iz_three_halves = Operator.Iz(1.5)
        assert np.allclose(
            Iz_three_halves.matrix,
            0.5 * np.array([
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, -3.0],
            ], dtype="complex"),
            rtol=0.0,
            atol=1E-10,
        )

        Ix_five_halves = Operator.Ix(2.5)
        assert np.allclose(
            Ix_five_halves.matrix,
            0.5 * np.array([
                [0.0, sqrt5, 0.0, 0.0, 0.0, 0.0],
                [sqrt5, 0.0, 2 * sqrt2, 0.0, 0.0, 0.0],
                [0.0, 2 * sqrt2, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 2 * sqrt2, 0.0],
                [0.0, 0.0, 0.0, 2 * sqrt2, 0.0, sqrt5],
                [0.0, 0.0, 0.0, 0.0, sqrt5, 0.0],
            ], dtype="complex"),
            rtol=0.0,
            atol=1E-10,
        )

    def test_commutator(self):
        Ix = Operator.Ix(0.5)
        assert np.array_equal(
            Ix.matrix, np.array([[0.0, 0.5], [0.5, 0.0]], dtype="complex")
        )
        Iy = Operator.Iy(0.5)
        Iz = Operator.Iz(0.5)
        Iplus = Operator.Iplus(0.5)
        # [Iz, I+] = I+
        assert Iz.commutator(Iplus) == Iplus
        # [Ix, Iy] = iIz
        assert Ix.commutator(Iy) == 1j * Iz
        assert Iy.commutator(Ix) == -1j * Iz

    def test_matmul(self):
        Iz = Operator.Iz(0.5)
        Iz_squared = Operator(0.5, matrix=np.array([[0.25, 0.0], [0.0, 0.25]]))
        Iz_cubed = Operator(0.5, matrix=np.array([[0.125, 0.0], [0.0, -0.125]]))
        assert Iz @ Iz == Iz_squared
        assert Iz ** 2 == Iz_squared
        assert Iz @ Iz @ Iz == Iz_cubed
        assert Iz ** 3 == Iz_cubed

    def test_kroenecker(self):
        z = Operator.Iz(0.5)
        zz = 0.25 * np.diag(np.array([1, -1, -1, 1], dtype="complex"))
        assert z.kroenecker(z) == Operator(0.5, nspins=2, matrix=zz)

    def test_basis(self):
        basis = CartesianBasis(0.5, 2)
        Iz = basis.get("1z") + basis.get("2z")
