# pa.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 22 Feb 2022 10:02:03 GMT

"""Module for simulating 1D pulse-acquire experiments."""

from typing import Tuple, Union

import numpy as np
from numpy import fft

from nmr_sims.experiments import Simulation, SAMPLE_SPIN_SYSTEM
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem


class PulseAcquireSimulation(Simulation):
    """Simulation class for Pulse-acquire experiment."""
    dimension_number = 1
    channel_number = 1
    channel_mapping = [0]

    def __init__(
        self,
        spin_system: SpinSystem,
        points: int,
        sweep_width: Union[str, float, int],
        offset: Union[str, float, int] = 0.0,
        channel: Union[str, Nucleus] = "1H",
    ) -> None:
        """Initialise a simulaion object.

        Parameters
        ----------

        spin_system
            The spin system to perform the simulation on.

        points
            The number of points sampled.

        sweep_width
            The sweep width.

        offset
            The transmitter offset.

        channel
            The nucelus targeted by the channel.
        """
        super().__init__(spin_system, [points], [sweep_width], [offset], [channel])
        self.name = f"{self.channels[0].ssname} Pulse-Acquire"

    def _pulse_sequence(self) -> np.ndarray:
        pts, sw, off, nuc = (
            self.points[0], self.sweep_widths[0], self.offsets[0],
            self.channels[0].name,
        )

        # Hamiltonian propagator
        hamiltonian = self.spin_system.hamiltonian(offsets={nuc: off})
        evol = hamiltonian.rotation_operator(1 / sw)

        # Detection operator
        detect = self.spin_system.Ix(nuc) - 1j * self.spin_system.Iy(nuc)

        # Initialise density operator
        rho = self.spin_system.equilibrium_operator

        # Initialise FID array
        fid = np.zeros(pts, dtype="complex")

        # --- Apply π/2 pulse ---
        rho = rho.propagate(self.pulses[1]["y"]["90"])

        # --- Detection ---
        for i in range(pts):
            fid[i] = rho.expectation(detect)
            rho = rho.propagate(evol)

        fid *= np.exp(np.linspace(0, -10, pts))
        return fid

    def _fetch_fid(self) -> Tuple[np.ndarray, np.ndarray]:
        pts, sw = self.points[0], self.sweep_widths[0]
        tp = np.linspace(0, (pts - 1) / sw, pts)
        return tp, self._fid

    def _fetch_spectrum(self, zf_factor: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        sw, off, pts, sfo = (
            self.sweep_widths[0], self.offsets[0], self.points[0], self.sfo[0],
        )
        shifts = np.linspace((sw / 2) + off, -(sw / 2) + off, pts * zf_factor) / sfo
        spectrum = fft.fftshift(
            fft.fft(
                self._fid,
                pts * zf_factor,
            )
        )

        return shifts, np.flip(spectrum)


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("tkAgg")

    # AX3 1H spin system with A @ 2ppm and X @ 7ppm.
    # Field of 500MHz
    spin_system = SAMPLE_SPIN_SYSTEM

    # Experiment parameters
    channel = "1H"
    sweep_width = "10ppm"
    points = 8192
    offset = "5ppm"

    # Simulate the experiment
    sim = PulseAcquireSimulation(spin_system, points, sweep_width, offset, channel)
    sim.simulate()
    # Extract FID and timepoints
    tp, fid = sim.fid()
    # Extract spectrum and chemical shifts
    shifts, spectrum = sim.spectrum(zf_factor=4)

    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(tp, np.real(fid))
    axs[1].plot(shifts, np.real(spectrum))
    axs[1].set_xlim(reversed(axs[1].get_xlim()))
    plt.show()
