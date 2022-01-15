# _operators.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sat 15 Jan 2022 20:07:05 GMT

from __future__ import annotations
import re
from typing import Any, Union
import numpy as np
import numpy.linalg as nlinalg
import scipy.linalg as slinalg

from nmr_sims import _sanity


class Operator:
    def __init__(self, matrix: np.ndarray):
        if not isinstance(matrix, np.ndarray):
            raise TypeError("`matrix` should be a NumPy array.")
        shape = matrix.shape
        if len(shape) == 2 and (shape[0] == shape[1]):
            self.matrix = matrix.astype("complex")
        else:
            raise ValueError("`matrix` should be a square 2D array.")

    def __str__(self) -> str:
        return str(self.matrix)

    def __eq__(self, other: Operator) -> bool:
        self._check_other_is_same_dim_operator(other)
        return np.allclose(self.matrix, other.matrix, rtol=0, atol=1E-10)

    def __neg__(self) -> Operator:
        return Operator(-self.matrix)

    def __add__(self, other: Operator) -> Operator:
        self._check_other_is_same_dim_operator(other)
        return Operator(self.matrix + other.matrix)

    def __sub__(self, other: Operator) -> Operator:
        self._check_other_is_same_dim_operator(other)
        return Operator(self.matrix - other.matrix)

    def __mul__(self, other: Union[int, float, complex]) -> Operator:
        if isinstance(other, (int, float, complex)):
            return Operator(self.matrix * other)
        else:
            raise TypeError(f"{other} must be a scalar.")

    def __rmul__(self, other: Union[int, float, complex]) -> Operator:
        if isinstance(other, (int, float, complex)):
            return Operator(other * self.matrix)
        else:
            raise TypeError(f"{other} must be a scalar.")

    def __truediv__(self, other: Union[int, float, complex]) -> Operator:
        if isinstance(other, (int, float, complex)):
            return Operator(self.matrix / other)
        else:
            raise TypeError(f"{other} must be a scalar.")

    def __matmul__(self, other: Union[np.ndarray, Operator]) -> Operator:
        if isinstance(other, Operator):
            return Operator(self.matrix @ other.matrix)
        else:
            raise TypeError(f"{other} must be an `Operator`.")

    def __pow__(self, power: Union[int, float, complex]) -> Operator:
        if isinstance(power, (int, float, complex)):
            return Operator(nlinalg.matrix_power(self.matrix, power))
        else:
            raise TypeError("`power` should be a scalar.")

    @property
    def dim(self):
        return self.matrix.shape[0]

    @property
    def adjoint(self) -> Operator:
        return Operator(self.matrix.conj().T)

    @property
    def trace(self) -> float:
        return np.einsum('ii', self.matrix)

    @property
    def exp(self) -> Operator:
        return Operator(slinalg.expm(self.matrix))

    def expectation(self, other: Operator) -> float:
        self._check_other_is_same_dim_operator(other)
        return np.einsum("ij,ij->", other.matrix.T, self.matrix)

    def commutator(self, other: Operator) -> Operator:
        self._check_other_is_same_dim_operator(other)
        return (self @ other) - (other @ self)

    def commutes_with(self, other: Operator) -> bool:
        self._check_other_is_same_dim_operator(other)
        return self.commutator(other) == Operator(np.zeros((self.dim, self.dim)))

    def kroenecker(self, other: Operator) -> Operator:
        self._check_other_is_operator(other)
        return Operator(np.kron(self.matrix, other.matrix))

    def rotation_operator(self, angle: float) -> Operator:
        if isinstance(angle, (int, float)):
            return Operator(slinalg.expm(-1j * angle * self.matrix))
        else:
            raise TypeError("`angle` should be a scalar.")

    def propagate(self, propagator: Operator) -> Operator:
        self._check_other_is_same_dim_operator(propagator)
        return propagator @ self @ propagator.adjoint

    def _check_other_is_operator(self, other: Any) -> None:
        if not isinstance(other, Operator):
            raise TypeError(f"{other} should be an `Operator`.")

    def _check_other_is_same_dim_operator(self, other: Any) -> None:
        if not isinstance(other, Operator):
            raise TypeError(f"{other} should be an `Operator`.")
        elif not self.matrix.shape == other.matrix.shape:
            raise ValueError(
                f"Operator dimension should match, but are{self.matrix.shape}"
                f"and {other.matrix.shape}."
            )

    @staticmethod
    def _check_I_is_a_multiple_of_one_half(I: Union[int, float]) -> None:
        if not _sanity.is_multiple_of_one_half(I):
            raise ValueError(f"`I` should be a multiple of 1/2, but is {I}")

    @classmethod
    def Ix(cls, I: Union[int, float]) -> Operator:
        cls._check_I_is_a_multiple_of_one_half(I)
        dim = int(2 * I + 1)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[cls._diagonal_indices(dim, k=1)] = 0.5 * np.sqrt(
            [I * (I + 1) - m * (m + 1) for m in np.linspace(-I, I - 1, dim - 1)]
        )
        matrix[cls._diagonal_indices(dim, k=-1)] = 0.5 * np.sqrt(
            [I * (I + 1) - m * (m - 1) for m in np.linspace(-I + 1, I, dim - 1)]
        )
        return cls(matrix)

    @classmethod
    def Iy(cls, I: Union[int, float]) -> Operator:
        cls._check_I_is_a_multiple_of_one_half(I)
        dim = int(2 * I + 1)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[cls._diagonal_indices(dim, k=1)] = -0.5j * np.sqrt(
            [I * (I + 1) - m * (m + 1) for m in np.linspace(-I, I - 1, dim - 1)]
        )
        matrix[cls._diagonal_indices(dim, k=-1)] = 0.5j * np.sqrt(
            [I * (I + 1) - m * (m - 1) for m in np.linspace(-I + 1, I, dim - 1)]
        )
        return cls(matrix)

    @classmethod
    def Iz(cls, I: Union[int, float]) -> Operator:
        cls._check_I_is_a_multiple_of_one_half(I)
        dim = int(2 * I + 1)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[cls._diagonal_indices(dim)] = np.round(np.linspace(I, -I, dim), 1)
        return cls(matrix)

    @classmethod
    def E(cls, I: Union[int, float]) -> Operator:
        # TODO: Constant factor depending on I?
        cls._check_I_is_a_multiple_of_one_half(I)
        dim = int(2 * I + 1)
        return cls(0.5 * np.eye(dim, dtype="complex"))

    @classmethod
    def Iplus(cls, I: Union[int, float]) -> Operator:
        cls._check_I_is_a_multiple_of_one_half(I)
        dim = int(2 * I + 1)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[cls._diagonal_indices(dim, k=1)] = np.sqrt(
            [I * (I + 1) - m * (m + 1) for m in np.linspace(-I, I - 1, dim - 1)]
        )
        return cls(matrix)

    @classmethod
    def Iminus(cls, I: Union[int, float]) -> Operator:
        dim = int(2 * I + 1)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[cls._diagonal_indices(dim, k=-1)] = np.sqrt(
            [I * (I + 1) - m * (m - 1) for m in np.linspace(-I + 1, I, dim - 1)]
        )
        return cls(matrix)

    @staticmethod
    def _diagonal_indices(size: int, k: int = 0):
        rows, cols = np.diag_indices(size)
        if k < 0:
            return rows[-k:], cols[:k]
        elif k > 0:
            return rows[:-k], cols[k:]
        else:
            return rows, cols


class CartesianBasis:
    def __init__(self, *, I: float = 0.5, nspins: int = 1) -> None:
        if not (isinstance(I, (int, float)) and _sanity.is_multiple_of_one_half(I)):
            raise ValueError(f"`I` should be a multiple of 1/2, but is {I}.")
        if not (isinstance(nspins, int) and nspins > 0):
            raise ValueError(
                f"`nspins` should be an int greater than 0 but is {nspins}."
            )
        self._Ix = Operator.Ix(I)
        self._Iy = Operator.Iy(I)
        self._Iz = Operator.Iz(I)
        self._E = Operator.E(I)
        self.nspins = nspins
        self.I = I

    @property
    def dim(self):
        return int(2 * self.I + 1) ** self.nspins

    @property
    def zero(self) -> np.ndarray:
        return Operator(np.zeros((self.dim, self.dim)))

    @property
    def identity(self) -> np.ndarray:
        return Operator(np.eye(self.dim))

    def get(self, operator: str) -> Operator:
        if not isinstance(operator, str):
            raise ValueError("`operator` should be a str.")
        err_preamble = f"`operator` is invalid: \"{operator}\"\n"

        full_regex = r"^(\d+(x|y|z))+$"
        component_regex = r"\d+(?:x|y|z)"
        if re.match(full_regex, operator):
            elements = {}
            for component in re.findall(component_regex, operator):
                num = int(re.search(r"\d+", component).group(0))
                coord = re.search(r"(x|y|z)", component).group(0)
                if num > self.nspins:
                    raise ValueError(
                        f"{err_preamble}Spin {num} does not exist for basis of "
                        f"{self.nspins} spins."
                    )
                if num in elements:
                    raise ValueError(
                        f"{err_preamble}Spin {num} is repeated."
                    )
                elements[num] = coord
        elif operator == "e":
            elements = {}
        else:
            raise ValueError(
                f"{err_preamble}Should satisfy the regex {full_regex}"
            )

        operator = Operator(np.array([[1]]))
        for i in range(1, self.nspins + 1):
            if i in elements:
                coord = elements[i]
                operator = operator.kroenecker(self.__dict__[f"_I{coord}"])
            else:
                operator = operator.kroenecker(self._E)

        return 2 ** (self.nspins - 1) * operator

    def _get_sum(self, coord: str) -> Operator:
        attr = f"_I{coord}_sum"
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            sum_ = sum(
                [self.get(f"{i}{coord}") for i in range(1, self.nspins + 1)],
                start=self.zero
            )
            setattr(self, attr, sum_)
            return sum_

    @property
    def Ix(self) -> Operator:
        return self._get_sum("x")

    @property
    def Iy(self) -> Operator:
        return self._get_sum("y")

    @property
    def Iz(self) -> Operator:
        return self._get_sum("z")
