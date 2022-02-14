# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 14 Feb 2022 11:44:04 GMT

from dataclasses import dataclass
from typing import Dict, Iterable, Union
import numpy as np
from nmr_sims.spin_system import SpinSystem


@dataclass
class Result:
    _fid: Dict[str, np.ndarray]
    _dim_info: Iterable[Dict]
    _field: float

    @property
    def dim(self):
        return len(self._dim_info)

    def fid(
        self, component: Union[str, None] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        raise AttributeError("`fid` has not been defined for this class!")

    def spectrum(self, *args, **kwargs):
        raise AttributeError("`sprectrum` has not been defined for this class!")

    @property
    def pts(self):
        return [di["pts"] for di in self._dim_info]

    def sw(self, unit: str = "hz"):
        if unit == "hz":
            return [di["sw"] for di in self._dim_info]
        elif unit == "ppm":
            return [di["sw"] / (1e-6 * di["nuc"].gamma / (2 * np.pi) * self._field)
                    for di in self._dim_info]

    def offset(self, unit: str = "hz"):
        if unit == "hz":
            return [di["off"] for di in self._dim_info]
        elif unit == "ppm":
            return [di["off"] / (1e-6 * di["nuc"].gamma / (2 * np.pi) * self._field)
                    for di in self._dim_info]


SAMPLE_SPIN_SYSTEM = SpinSystem(
    {
        1: {
            "shift": 2,
            "couplings": {
                2: 10,
                3: 10,
                4: 10,
            },
        },
        2: {
            "shift": 7,
        },
        3: {
            "shift": 7,
        },
        4: {
            "shift": 7,
        },
    },
)
