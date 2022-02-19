# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 18 Feb 2022 17:20:29 GMT

"""This module contains a collection of pre-defined functions for simulating
a number of solution-state NMR experiments. The current available experiments are:

* 1D Pulse-acquire experiment (:py:mod:`nmr_sims.experiments.pa`)
* Homonuclear J-resolved experiment. (:py:mod:`nmr_sims.experiments.jres`)

As well as this, there are contructs to help create functions for custom
experiment simulations.
"""

from __future__ import annotations

from typing import Dict, Iterable, Union, Tuple
import numpy as np

from nmr_sims import _sanity
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem


def copydoc(fromfunc, sep="\n"):
    """Decorator: copy the docstring of `fromfunc`."""
    def _decorator(func):
        sourcedoc = fromfunc.__doc__
        if func.__doc__ is None:
            func.__doc__ = sourcedoc
        else:
            func.__doc__ = sep.join([sourcedoc, func.__doc__])
        return func
    return _decorator


class Simulation:
    """Abstract simulation base class."""
    dimension_number = None
    dimension_labels = []
    channel_number = None
    name = "unknown"

    def __init__(
        self,
        spin_system: SpinSystem,
        points: Tuple[int],
        sweep_widths: Tuple[Union[str, float, int]],
        offsets: Tuple[Union[str, float, int]],
        channels: Tuple[Union[str, Nucleus]],
    ) -> None:
        """Initialise an instance of the class.

        Parameters
        ----------

        spin_system
            The spin system to perform the simulation on.

        points
            The number of points sampled in each dimension.

        sweep_widths
            The sweep widths in each dimension.

        offsets
            The transmitter offsets for each channel

        channels
            The nucelus targeted with each channel.
        """
        self.__dict__.update(locals())
        self.process_params()
        self._fid = None

    def process_params(self) -> None:
        """Process experiment simulation parameters.

        If any inputs given by the user are faulty, an error will be raised.
        Otherwise, correctly processed values for the channels, sweep widths, offsets
        and number of points will be stored.
        """
        self._check_right_length("points")
        self._check_right_length("sweep_widths")
        self._check_right_length("offsets")
        self._check_right_length("channels")

        self.points = [_sanity.process_points(p) for p in self.points]
        self.channels = [_sanity.process_nucleus(c, None) for c in self.channels]
        self.sweep_widths = [
            _sanity.process_sweep_width(
                sw, self.channels[self.channel_mapping[i]], self.spin_system.field
            )
            for i, sw in enumerate(self.sweep_widths)
        ]
        self.offsets = [
            _sanity.process_offset(offset, channel, self.spin_system.field)
            for offset, channel in zip(self.offsets, self.channels)
        ]

    def _check_right_length(self, name: str) -> None:
        if name in ["points", "sweep_widths"]:
            length = self.dimension_number
        elif name in ["channels", "offsets"]:
            length = self.channel_number
        if len(getattr(self, name)) != length:
            raise ValueError(f"`{name}` should be an iterable of length {length}.")

    def log(self) -> None:
        swstr = ", ".join(
            [f"{x:.3f} (F{i})" for i, x in enumerate(self.sweep_widths, start=1)]
        )
        channelstr = "\n".join(
            [f"* Channel {i}: {nuc.ssname}, offset: {off:.3f} Hz"
             for i, (nuc, off) in enumerate(zip(self.channels, self.offsets), start=1)]
        )
        ptsstr = ", ".join(
            [f"{x} (F{i})" for i, x in enumerate(self.points, start=1)]
        )
        msg = f"Simulating {self.name} experiment"
        msg += f"\n{len(msg) * '-'}\n"
        msg += (
            f"* Temperature: {self.spin_system.temperature} K\n"
            f"* Field Strength: {self.spin_system.field} T\n"
            f"* Sweep width: {swstr}\n{channelstr}\n"
            f"* Points sampled: {ptsstr}"
        )
        print(msg)

    def simulate(self) -> Result:
        """Simulate the NMR experiment."""
        self.log()
        self._fid = self._pulse_sequence()

    def _pulse_sequence(self) -> np.ndarray:
        """Definition of the pulse sequence."""
        raise AttributeError(
            "`pulse_sequence` needs to be defined!"
        )

    def check_if_fid_is_none(f: callable) -> callable:
        def wrapper(*args, **kwargs):
            # The first arg is the class instance.
            if args[0].fid is None:
                raise ValueError(
                    "No FID is associated with the simulation. Perhaps you need to "
                    "call `simulate()` before trying to access it?"
                )
            return f(*args, **kwargs)
        return wrapper

    @check_if_fid_is_none
    def fid(self) -> np.ndarray:
        """Return the timepoints sampled and the FID generated."""
        if getattr(self, "_fetch_fid", None) is None:
            raise AttributeError(
                "BUG: `_fetch_fid` needs to be defined!"
            )
        else:
            return self._fetch_fid()

    @check_if_fid_is_none
    def spectrum(self, zf_factor=1) -> Tuple[np.ndarray, np.ndarray]:
        """Return the chemical shifts and spectrum.

        Parameters
        ----------

        zf_factor
            The ratio between the number of points in the final spectrum,
            generated by zero-filling the FID, and the FID itself. ``1``
            (default) means no zero-filling is applied.
        """
        if getattr(self, "_fetch_spectrum", None) is None:
            raise AttributeError(
                "BUG: `_fetch_sprectrum` needs to be defined!"
            )
        else:
            return self._fetch_spectrum(zf_factor)



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
