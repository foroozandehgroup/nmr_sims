# spin_system.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 13 Jan 2022 18:07:23 GMT

from typing import Any, Iterable, Tuple, Union
import numpy as np
from nmr_sims.experimental import Experimental
from nmr_sims.nuclei import Nucleus
from nmr_sims._operators import CartesianBasis, Operator
from nmr_sims import _sanity


# Currently only applicable for homonuclear systems
class SpinSystem(CartesianBasis):
    def __init__(
        self, spins: dict, nucleus: Union[str, Nucleus] = "1H",
    ) -> None:
        self.nucleus = _sanity.process_nucleus(nucleus)
        self.shifts, self.couplings = self._process_spins(spins)
        super().__init__(I=self.nucleus.spin, nspins=self.shifts.size)
        self.experimental = None
        self._conditions_set = False

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

    def set_conditions(self, experimental: Experimental) -> None:
        self.experimental = experimental
        self._conditions_set = True

    def _check_conditions_set(self) -> None:
        if not self._conditions_set:
            raise ValueError("`set_conditions` must be applied.")

    def pulse(self, phase: float = 0., angle: float = np.pi / 2) -> Operator:
        operator = self.Ix * np.cos(phase) + self.Iy * np.sin(phase)
        return operator.rotation_operator(angle)

    @property
    def frequencies(self) -> Iterable[float]:
        self._check_conditions_set()
        channels = self.experimental.channels
        channel = self.experimental.channels[len(channels)]
        gamma = channel.nucleus.gamma / (2 * np.pi)
        return 2 * np.pi * np.array([
            ((1e-6 * gamma * self.experimental.field * shift) - channel.offset)
            for shift in self.shifts
        ])

    @property
    def equilibrium_operator(self) -> Operator:
        self._check_conditions_set()
        B = self.experimental.boltzmann_factor[0]
        return (1 / self.nspins) * (self.identity + (B * self.Iz))

    @property
    def hamiltonian(self) -> Operator:
        self._check_conditions_set()
        freqs = self.frequencies
        couplings = self.couplings
        H = self.zero
        for i in range(self.nspins):
            H += freqs[i] * self.get(f"{i + 1}z")
            for j in range(i + 1, self.nspins):
                H += np.pi * couplings[i, j] * (
                    self.get(f"{i + 1}x{j + 1}x") +
                    self.get(f"{i + 1}y{j + 1}y") +
                    self.get(f"{i + 1}z{j + 1}z")
                )

        return H

    @staticmethod
    def _process_spins(spins: Any) -> Tuple[np.ndarray, np.ndarray]:
        _sanity.check_dict_with_int_keys(spins, "spins", consecutive=True)
        nspins = len(spins)
        shifts = np.zeros(nspins)
        couplings = np.zeros((nspins, nspins))

        for i, spin in spins.items():
            if "shift" not in spin.keys():
                raise ValueError(
                    "Each value in `spins` should be a dict with the keys "
                    "\"shift\" and (optional) \"couplings\". "
                    f"This is not satisfied by spin {i}."
                )

            if not isinstance(spin["shift"], (int, float)):
                raise TypeError(
                    "\"shift\" entries should be scalar values. This is not "
                    f"satisfied by spin {i}."
                )

            shifts[i - 1] = spin["shift"]

            if "couplings" in spin.keys():
                _sanity.check_dict_with_int_keys(
                    spin["couplings"], f"spins[{i}][\"couplings\"]", max_=nspins,
                    forbidden=[i],
                )

                for j, coupling in spin["couplings"].items():
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

    def _check_valid_spin(self, spin: Any) -> None:
        if not (isinstance(spin, int) and 1 <= spin <= self.nspins):
            raise ValueError(
                "`spin` should be an int greater than 0 and no more than "
                f"{self.nspins}"
            )
