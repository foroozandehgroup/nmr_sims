# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 16 Feb 2022 11:55:10 GMT

"""This module contains a collection of pre-defined functions for simulating
a number of solution-state NMR experiments. The current available experiments are:

* 1D Pulse-acquire experiment (:py:mod:`nmr_sims.experiments.pa`)
* Homonuclear J-resolved experiment. (:py:mod:`nmr_sims.experiments.jres`)

As well as this, there are contructs to help create functions for custom
experiment simulations.
"""

from typing import Dict, Iterable, Union, Tuple
import numpy as np

from nmr_sims import _sanity
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem


class Result:
    """Object representing the resulting dataset derived from an experiment
    simulation.

    This is a base class. For each experiment function, there should be a
    corresponding result class which inherits from ``Result``, which defines the
    methods ``fid`` and ``spectrum`` for the specific experiment. See for example
    :py:class:`nmr_sims.experiments.pa.PulseAcquireResult` or
    :py:class:`nmr_sims.experiments.jres.JresResult`.
    """

    def __init__(
        self,
        fid: Dict[str, np.ndarray],
        dim_info: Iterable[Dict],
        spin_system: SpinSystem,
    ) -> None:
        """Generate a result instant.

        Parameters
        ----------

        fid
            The time domain signal(s) obtained from the simulation.

        dim_info
            Information on experiment parameters in each dimension of the
            experiment. For each dimension ``dict`` there should be the following
            entries:

            * ``"nuc"`` - :py:class:`nmr_sims.nuclei.Nucleus`, denoting the
              identity of the channel in the dimension.
            * ``"sw"`` - ``float``, the sweep width in the dimension, in Hz.
            * ``"off"`` - ``float``, the transmitter offset in the dimension, in Hz.
            * ``"pts"`` - ``int``, the number of points sampled in the dimension.

        spin_system
            The spin system object given to the simulation function.
        """
        self._fid = fid
        self._dim_info = dim_info
        self._spin_system = spin_system

    @property
    def dim(self) -> int:
        """Return the number of dimensions in the data."""
        return len(self._dim_info)

    def fid(
        self, component: Union[str, None] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        raise AttributeError("`fid` has not been defined for this class!")

    def spectrum(self, *args, **kwargs):
        raise AttributeError("`sprectrum` has not been defined for this class!")

    @property
    def pts(self) -> Iterable[int]:
        """Return the number of points sampled in each dimension."""
        return [di["pts"] for di in self._dim_info]

    def sw(self, unit: str = "hz") -> Iterable[float]:
        """Return the sweep width in each dimension.

        Parameters
        ----------

        unit
            Should be either ``"hz"`` or ``"ppm"``.
        """
        if unit == "hz":
            return [di["sw"] for di in self._dim_info]
        elif unit == "ppm":
            return [di["sw"] / (1e-6 * di["nuc"].gamma / (2 * np.pi) *
                                self._spin_system.field)
                    for di in self._dim_info]

    def offset(self, unit: str = "hz"):
        """Return the transmitter offset in each dimension.

        Parameters
        ----------

        unit
            Should be either ``"hz"`` or ``"ppm"``.
        """
        if unit == "hz":
            return [di["off"] for di in self._dim_info]
        elif unit == "ppm":
            return [di["off"] / (1e-6 * di["nuc"].gamma / (2 * np.pi) *
                                 self._spin_system.field)
                    for di in self._dim_info]

    @property
    def nuclei(self) -> Iterable[Nucleus]:
        """Return the targeted nucleus (channel) for each dimension."""
        return [di["nuc"] for di in self._dim_info]


def _check_right_length(obj: Iterable, name: str, length: int) -> None:
    if len(obj) != length:
        if length == 1 and name != "points":
            name = name[:-1]
        raise ValueError(f"`{name}` should be an iterable of length {length}.")


def process_params(
    dimension_number: int,
    channel_number: int,
    channel_mapping: Iterable[int],
    points: Iterable[int],
    sweep_widths: Iterable[Union[str, float, int]],
    offsets: Iterable[Union[str, float, int]],
    channels: Iterable[Nucleus],
    field: float,
) -> Tuple[Nucleus, Tuple[float, float], float, Tuple[int, int]]:
    """Process experiment simulation parameters.

    If any inputs given by the user are faulty, an error will be raised.
    Otherwise, correctly processed values for the channels, sweep widths, offsets
    and number of points will be returned.

    Parameters
    ----------

    dimension_number
        The number of data dimensions.

    channel_number
        The number of channels used in the simulation.

    channel_mapping
        A list of length ``dimension_number`` which inidcates the channel associated
        with each data dimension.

    points
        The points in each dimension.

    sweep_widths
        The sweep widths in each dimension.

    offsets
        The transmitter offsets associated with each channel.

    channels
        The identities of each channel.
    """
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


#: ``SAMPLE_SPIN_SYSTEM`` corresponds to a simple AX₃ spin system, with:
#:
#: .. math::
#:
#:     \delta_A = 2\ \mathrm{ppm},\ \delta_X = 7\ \mathrm{ppm},
#:     \ J_{AX} = 10\ \mathrm{Hz}
#:
#:     T = 298\ \mathrm{K},\ B_0 = 500\ \mathrm{MHz}\ (\approx 11.74\ \mathrm{T})
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
