# spin_system.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sun 16 Jan 2022 16:04:59 GMT

from typing import Iterable, Union
import numpy as np
from nmr_sims.nuclei import Nucleus
from nmr_sims._operators import CartesianBasis, Operator
from nmr_sims import _sanity


# Currently only applicable for homonuclear systems
class SpinSystem(CartesianBasis):
    boltzmann = 1.380649e-23
    hbar = 1.054571817e-34

    def __init__(
        self, spins: dict, default_nucleus: Union[Nucleus, str] = "1H",
        field: Union[int, float, str] = "500MHz",
        temperature: Union[int, float, str] = "298K",
    ) -> None:
        self.temperature = _sanity.process_temperature(temperature)
        self.field = _sanity.process_field(field)
        self.spin_dict = _sanity.process_spins(spins, default_nucleus)
        super().__init__(
            [(spin.nucleus.multiplicity - 1) / 2 for spin in self.spin_dict.values()]
        )
        if not all([I == 0.5 for I in self.spins]):
            raise ValueError("Spin-1/2 nuclei are only supported currently!")

    @property
    def inverse_temperature(self):
        return self.hbar / (self.boltzmann * self.temperature)

    @property
    def boltzmann_factor(self) -> Iterable[float]:
        return [
            2 * np.pi * self.inverse_temperature * spin.nucleus.gamma * self.field
            for spin in self.spin_dict.values()
        ]

    @property
    def basic_frequencies(self) -> Iterable[float]:
        return [
            -spin.nucleus.gamma * self.field * (1 + (1e-6 * spin.shift))
            for spin in self.spin_dict.values()
        ]

    @property
    def rotframe_frequencies(self) -> Iterable[float]:
        return [
            -spin.nucleus.gamma * self.field * (1e-6 * spin.shift)
            for spin in self.spin_dict.values()
        ]

    def pulse(
        self, nucleus: str, phase: float = 0., angle: float = np.pi / 2
    ) -> Operator:
        operator = self.zero
        for i, spin in self.spin_dict.items():
            if spin.nucleus.name == nucleus:
                operator += (
                    np.cos(phase) * self.get(f"{i}x") +
                    np.sin(phase) * self.get(f"{i}y")
                )
        return operator.rotation_operator(angle)

    @property
    def equilibrium_operator(self) -> Operator:
        return (1 / self.nspins) * (
            self.identity + sum(
                [b * self.get(f"{i}z") for i, b in
                 enumerate(self.boltzmann_factor, start=1)],
                start=self.zero,
            )
        )

    def _get_frequency(self, label: int):
        spin = self.spin_dict[label]
        return -spin.nucleus.gamma * self.field * (1e-6 * spin.shift)

    def _get_coupling(self, label1: int, label2: int) -> float:
        label1, label2 = (min((label1, label2)), max((label1, label2)))
        return self.spin_dict[label1].couplings[label2]

    def hamiltonian(self, offsets: Union[dict, None] = None) -> Operator:
        H = self.zero
        for i in range(1, self.nspins + 1):
            H += self._get_frequency(i) * self.get(f"{i}z")
            if i == self.nspins:
                break
            for j in range(i + 1, self.nspins + 1):
                H += 2 * np.pi * self._get_coupling(i, j) * (
                    self.get(f"{i}x{j}x") +
                    self.get(f"{i}y{j}y") +
                    self.get(f"{i}z{j}z")
                )

        if offsets is None:
            pass
        else:
            for nuc, off in offsets.items():
                H += 2 * np.pi * off * sum(
                    [self.get(f"{i}z")
                     for i, spin in self.spin_dict.items()
                     if spin.nucleus.name == nuc],
                    start=self.zero
                )

        return H

    def _get_sum(self, coord: str, nucleus: str) -> Operator:
        if nucleus is None:
            labels = list(self.spin_dict.keys())
        else:
            labels = [i for i, spin in self.spin_dict.items()
                      if spin.nucleus.name == nucleus]

        return sum([self.get(f"{i}{coord}") for i in labels], start=self.zero)

    def Ix(self, nucleus: Union[str, None] = None) -> Operator:
        return self._get_sum("x", nucleus)

    def Iy(self, nucleus: Union[str, None] = None) -> Operator:
        return self._get_sum("y", nucleus)

    def Iz(self, nucleus: Union[str, None] = None) -> Operator:
        return self._get_sum("z", nucleus)
