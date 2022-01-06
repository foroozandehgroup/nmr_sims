# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 20 Dec 2021 23:02:15 GMT

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from nmr_sims import CartesianBasis, Operator


def get_spin_system(spin_system: Dict[int, Dict]) -> Tuple[np.ndarray, np.ndarray]:
    ss_keys = list(spin_system.keys())
    nspins = len(ss_keys)
    spin_labels = list(range(1, nspins + 1))
    assert all([isinstance(key, int) for key in ss_keys])
    assert sorted(ss_keys) == spin_labels

    shifts = np.zeros(nspins)
    couplings = np.zeros((nspins, nspins))
    for i, spin in spin_system.items():
        assert all([k in spin for k in ("shift", "couplings")])
        assert isinstance(spin["shift"], (int, float))
        shifts[i - 1] = spin["shift"]

        j_keys = spin["couplings"].keys()
        assert all([isinstance(k, int) for k in j_keys])
        assert all([0 < k <= nspins for k in j_keys])
        for j, J in spin["couplings"].items():
            current = couplings[i - 1, j - 1]
            if current != 0.:
                assert J == current
            else:
                couplings[i - 1, j - 1] = J
                couplings[j - 1, i - 1] = J

    return SpinSystem(0.5, shifts, couplings)


class SpinSystem(CartesianBasis):
    def __init__(self, I: float, shifts: np.ndarray, couplings: np.ndarray):
        nspins = shifts.size
        super().__init__(I, nspins)
        self.shifts = shifts
        self.couplings = couplings

    @property
    def free_hamiltonian(self):
        H = Operator(self.I, self.nspins)
        for i in range(self.nspins):
            H += 2 * np.pi * self.shifts[i] * self.get(f"{i + 1}z")
            for j in range(i + 1, self.nspins):
                H += 2 * np.pi * self.couplings[i, j] * self.get(f"{i + 1}z{j + 1}z")

        return H
