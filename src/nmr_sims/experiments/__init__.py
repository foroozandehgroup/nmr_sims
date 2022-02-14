# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 14 Feb 2022 16:55:31 GMT

from dataclasses import dataclass
from typing import Dict, Iterable, Union, Tuple
import numpy as np

from nmr_sims import _sanity
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem


@dataclass
class Result:
    _fid: Dict[str, np.ndarray]
    _dim_info: Iterable[Dict]
    _spin_system: SpinSystem

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
            return [di["sw"] / (1e-6 * di["nuc"].gamma / (2 * np.pi) *
                                self._spin_system.field)
                    for di in self._dim_info]

    def offset(self, unit: str = "hz"):
        if unit == "hz":
            return [di["off"] for di in self._dim_info]
        elif unit == "ppm":
            return [di["off"] / (1e-6 * di["nuc"].gamma / (2 * np.pi) *
                                 self._spin_system.field)
                    for di in self._dim_info]

    @property
    def nuclei(self):
        return [di["nuc"] for di in self._dim_info]


def _check_right_length(obj: Iterable, name: str, length: int) -> None:
    if len(obj) != length:
        if length == 1 and name != "points":
            name = name[:-1]
        raise ValueError(f"`{name}` should be an iterable of length {length}.")


def _process_params(
    dimension_number: int,
    channel_number: int,
    channel_mapping: Iterable[int],
    points: Iterable[int],
    sweep_widths: Iterable[Union[str, float, int]],
    offsets: Iterable[Union[str, float, int]],
    channels: Iterable[Nucleus],
    field: float,
) -> Tuple[Nucleus, Tuple[float, float], float, Tuple[int, int]]:
    _check_right_length(points, "points", dimension_number)
    _check_right_length(sweep_widths, "sweep_widths", dimension_number)
    _check_right_length(offsets, "offsets", channel_number)
    _check_right_length(channels, "channels", channel_number)

    points = [_sanity.process_points(p) for p in points]
    channels = [_sanity.process_nucleus(c, None) for c in channels]
    sweep_widths = [
        _sanity.process_sweep_width(sw, channels[channel_mapping[i]], field)
        for i, sw in enumerate(sweep_widths)
    ]
    offsets = [
        _sanity.process_offset(offset, channel, field)
        for offset, channel in zip(offsets, channels)
    ]

    return points, sweep_widths, offsets, channels


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
