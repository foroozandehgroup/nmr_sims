# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 28 Feb 2022 12:13:18 GMT

"""This module contains a collection of pre-defined objects for simulating
a number of solution-state NMR experiments. The current available experiments are:

* 1D Pulse-acquire (:py:mod:`nmr_sims.experiments.pa`)
* Homonuclear J-resolved (:py:mod:`nmr_sims.experiments.jres`)
* HSQC (:py:mod:`nmr_sims.experiments.hsqc`)

The :py:class:`Simulation` base class has been created with the intention of making
new oulse sequences easy to implement.
"""

from __future__ import annotations

from typing import Iterable, Union, Tuple
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
        self._process_params()
        self._generate_pulses()
        self._fid = None

    def _process_params(self) -> None:
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

    def _log(self) -> None:
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

    def _generate_pulses(self) -> None:
        halfpi = 0.5 * np.pi
        pi = np.pi
        threehalfpi = 1.5 * np.pi
        self.pulses = {}
        nucs = [channel.name for channel in self.channels]
        for i, nuc in enumerate(nucs, start=1):
            self.pulses[i] = {
                "x": {
                    "90": self.spin_system.pulse(nuc, phase=0., angle=halfpi),
                    "180": self.spin_system.pulse(nuc, phase=0., angle=pi),
                },
                "y": {
                    "90": self.spin_system.pulse(nuc, phase=halfpi, angle=halfpi),
                    "180": self.spin_system.pulse(nuc, phase=halfpi, angle=pi),
                },
                "-x": {
                    "90": self.spin_system.pulse(nuc, phase=pi, angle=halfpi),
                    "180": self.spin_system.pulse(nuc, phase=pi, angle=pi),
                },
                "-y": {
                    "90": self.spin_system.pulse(nuc, phase=threehalfpi, angle=halfpi),
                    "180": self.spin_system.pulse(nuc, phase=threehalfpi, angle=pi),
                },
            }

    @property
    def field(self) -> float:
        return self.spin_system.field

    @property
    def sfo(self) -> Iterable[float]:
        return [
            channel.gamma * self.field / (2e6 * np.pi)
            for channel in self.channels
        ]

    def simulate(self) -> None:
        """Simulate the NMR experiment."""
        self._log()
        self._fid = self._pulse_sequence()

    def _pulse_sequence(self) -> np.ndarray:
        """Definition of the pulse sequence."""
        raise AttributeError(
            "`pulse_sequence` needs to be defined!"
        )

    def _check_if_fid_is_none(self) -> None:
        if self._fid is None:
            raise ValueError(
                "No FID is associated with the simulation. Perhaps you need to "
                "call `simulate()` before trying to access it?"
            )

    @property
    def fid(self) -> np.ndarray:
        """Return the timepoints sampled and the FID generated."""
        self._check_if_fid_is_none()
        if getattr(self, "_fetch_fid", None) is None:
            raise AttributeError(
                "BUG: `_fetch_fid` needs to be defined!"
            )
        else:
            return self._fetch_fid()

    def spectrum(
        self, zf_factor=1
    ) -> Tuple[Iterable[np.ndarray], np.ndarray, Iterable[str]]:
        """Return the chemical shifts, spectrum, and axis labels.

        Parameters
        ----------

        zf_factor
            The ratio between the number of points in the final spectrum,
            generated by zero-filling the FID, and the FID itself. ``1``
            (default) means no zero-filling is applied.

        Returns
        -------

        shifts
            Chemical shifts in each dimension

        spectrum
            Processed spectrum

        labels
            Axis labels
        """
        self._check_if_fid_is_none()
        if getattr(self, "_fetch_spectrum", None) is None:
            raise AttributeError(
                "BUG: `_fetch_sprectrum` needs to be defined!"
            )
        else:
            return self._fetch_spectrum(zf_factor)


#: ``SAMPLE_SPIN_SYSTEM`` corresponds to a proton AX₃ spin system, with:
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
