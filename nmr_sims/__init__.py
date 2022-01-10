# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 10 Jan 2022 18:02:28 GMT

from __future__ import annotations
from pathlib import Path
import re
from typing import Any, Tuple, Union
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


def is_multiple_of_one_half(x):
    return round(x, 10) % 0.5 == 0


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
        if not is_multiple_of_one_half(I):
            raise ValueError(f"`I` should be a multiple of 1/2, but is {I}")

    @classmethod
    def Ix(cls, I: Union[int, float]) -> Operator:
        cls._check_I_is_a_multiple_of_one_half(I)
        dim = int(2 * I + 1)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[_diagonal_indices(dim, k=1)] = 0.5 * np.sqrt(
            [I * (I + 1) - m * (m + 1) for m in np.linspace(-I, I - 1, dim - 1)]
        )
        matrix[_diagonal_indices(dim, k=-1)] = 0.5 * np.sqrt(
            [I * (I + 1) - m * (m - 1) for m in np.linspace(-I + 1, I, dim - 1)]
        )
        return cls(matrix)

    @classmethod
    def Iy(cls, I: Union[int, float]) -> Operator:
        cls._check_I_is_a_multiple_of_one_half(I)
        dim = int(2 * I + 1)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[_diagonal_indices(dim, k=1)] = -0.5j * np.sqrt(
            [I * (I + 1) - m * (m + 1) for m in np.linspace(-I, I - 1, dim - 1)]
        )
        matrix[_diagonal_indices(dim, k=-1)] = 0.5j * np.sqrt(
            [I * (I + 1) - m * (m - 1) for m in np.linspace(-I + 1, I, dim - 1)]
        )
        return cls(matrix)

    @classmethod
    def Iz(cls, I: Union[int, float]) -> Operator:
        cls._check_I_is_a_multiple_of_one_half(I)
        dim = int(2 * I + 1)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[_diagonal_indices(dim)] = np.round(np.linspace(I, -I, dim), 1)
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
        matrix[_diagonal_indices(dim, k=1)] = np.sqrt(
            [I * (I + 1) - m * (m + 1) for m in np.linspace(-I, I - 1, dim - 1)]
        )
        return cls(matrix)

    @classmethod
    def Iminus(cls, I: Union[int, float]) -> Operator:
        dim = int(2 * I + 1)
        matrix = np.zeros((dim, dim), dtype="complex")
        matrix[_diagonal_indices(dim, k=-1)] = np.sqrt(
            [I * (I + 1) - m * (m - 1) for m in np.linspace(-I + 1, I, dim - 1)]
        )
        return cls(matrix)


class CartesianBasis:
    def __init__(self, *, I: float = 0.5, nspins: int = 1) -> None:
        if not (isinstance(I, (int, float)) and is_multiple_of_one_half(I)):
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

    def get(self, operator: str) -> Operator:
        err_preamble = f"`operator` is invalid: \"{operator}\"\n"
        if not (isinstance(operator, str) and re.match(r"^(\d+(x|y|z))+$", operator)):
            raise ValueError(
                f"{err_preamble}Should satisfy the regex ^(\\d+(x|y|z))+$"
            )
        elements = {}
        for component in re.findall(r"\d+(?:x|y|z)", operator):
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

        operator = Operator(np.array([[1]]))
        for i in range(1, self.nspins + 1):
            if i in elements:
                coord = elements[i]
                operator = operator.kroenecker(self.__dict__[f"_I{coord}"])
            else:
                operator = operator.kroenecker(self._E)

        return 2 ** (self.nspins - 1) * operator


class SpinSystem(CartesianBasis):
    # TODO Assuming spin-1/2 at the moment.
    def __init__(self, spins: dict) -> None:
        self.shifts, self.couplings = self._process_spins_dict(spins)
        super().__init__(I=0.5, nspins=self.shifts.size)

    @staticmethod
    def _process_spins_dict(spins: Any) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(spins, dict):
            raise TypeError("`spins` should be a dict.")

        spin_keys = list(spins.keys())
        nspins = len(spin_keys)
        if not all([isinstance(key, int) for key in spin_keys]):
            raise ValueError(
                "`spins` should solely have ints as keys."
            )

        if not sorted(spin_keys) == list(range(1, nspins + 1)):
            raise ValueError(
                "The keys in `spins` should be consecutive ints, starting at 1."
            )

        shifts = np.zeros(nspins)
        couplings = np.zeros((nspins, nspins))

        for i, spin in spins.items():
            if not all([k in spin for k in ("shift", "couplings")]):
                raise ValueError(
                    "Each value in `spins` should be a dict with the keys "
                    "\"shift\" and \"couplings\". "
                    f"This is not satisfied by spin {i}."
                )

            if not isinstance(spin["shift"], (int, float)):
                raise TypeError(
                    "\"shift\" entries should be scalar values. This is not "
                    f"satisfied by spin {i}."
                )

            shifts[i - 1] = spin["shift"]

            if not isinstance(spin["couplings"], dict):
                raise TypeError(
                    "\"couplings\" entries should be dicts. This is not satisfied by "
                    f"spin {i}."
                )

            for j, coupling in spin["couplings"].items():
                if not (isinstance(j, int) and (1 <= j <= nspins) and (j != i)):
                    raise ValueError(
                        f"Invalid key in couplings dict for spin {i}: {j}. Should "
                        f"be and int greater than 0, no more than {nspins}, and not "
                        f"{i}."
                    )

                current_value = couplings[i - 1, j - 1]
                if float(current_value) != 0.:
                    if coupling != current_value:
                        raise ValueError(
                            f"Contradictory couplings given between spins {j} and "
                            f"{i}: {float(coupling)} and {current_value}."
                        )
                else:
                    couplings[i - 1, j - 1] = coupling
                    couplings[j - 1, i - 1] = coupling

        return shifts, couplings

    def edit_shift(self, spin: int, value: Union[int, float]) -> None:
        self._check_valid_spin(spin)
        if not isinstance(value, (int, float)):
            raise TypeError("`value` should be a scalar value.")
        self.shifts[spin - 1] = value

    def edit_coupling(self, spin1: int, spin2: int, value: Union[int, float]) -> None:
        self._check_valid_spin(spin1)
        self._check_valid_spin(spin2)
        if spin1 == spin2:
            raise ValueError("`spin1` and `spin2` cannot match.")

        if not isinstance(value, (int, float)):
            raise TypeError("`value` should be a scalar value.")

        self.couplings[spin1 - 1, spin2 - 1] = value
        self.couplings[spin2 - 1, spin1 - 1] = value

    def _check_valid_spin(self, spin: Any) -> None:
        if not (isinstance(spin, int) and 1 <= spin <= self.nspins):
            raise ValueError(
                "`spin` should be an int greater than 0 and no more than "
                f"{self.nspins}"
            )

    @property
    def free_hamiltonian(self):
        H = Operator(np.zeros((self.dim, self.dim)))
        for i in range(self.nspins):
            H += 2 * np.pi * self.shifts[i] * self.get(f"{i + 1}z")
            for j in range(i + 1, self.nspins):
                H += 2 * np.pi * self.couplings[i, j] * (
                    self.get(f"{i + 1}x{j + 1}x") +
                    self.get(f"{i + 1}y{j + 1}y") +
                    self.get(f"{i + 1}z{j + 1}z")
                )

        return H
