# pa.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 18 Feb 2022 17:16:22 GMT

"""Module for simulating 1D pulse-acquire experiments."""

from typing import Tuple, Union

import numpy as np
from numpy import fft

from nmr_sims.experiments import copydoc, Result, Simulation, SAMPLE_SPIN_SYSTEM
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem


class PulseAcquireResult(Result):
    """Result object for pulse-acquire experiment simulation."""

    @copydoc(Result.__init__)
    def __init__(self, fid, dim_info, field):
        super().__init__(fid, dim_info, field)




class PulseAcquireSimulation(Simulation):
    """Simulation class for Pulse-acquire experiment."""
    dimesnion_number = 1
    channel_number = 1
    name = "Pulse-Acquire"

    @copydoc(Simulation.__init__)
    def __init__(
        self,
        spin_system: SpinSystem,
        points: Tuple[int],
        sweep_widths: Tuple[Union[str, float, int]],
        offsets: Tuple[Union[str, float, int]] = [0.0],
        channels: Tuple[Union[str, Nucleus]] = ["1H"],
    ) -> None:
        """
        Notes
        -----
        The expected lengths of each argument are:

        * points: 1
        * sweep_widths: 1
        * offsets: 1
        * channels: 1
        """
        super().__init__(spin_system, points, sweep_widths, offsets, channels)
        self.name = f"{self.channels[0].ssname} {self.name}"

    @copydoc(Simulation._pulse_sequence)
    def _pulse_sequence(self) -> np.ndarray:
        pts, sw, off, nuc = (
            self.points, self.sweep_widths, self.offsets, self.channels.name,
        )

        # Hamiltonian propagator
        hamiltonian = spin_system.hamiltonian(offsets={nuc: off})
        evol = hamiltonian.rotation_operator(1 / sw)

        # pi / 2 pulse propagator
        phase = np.pi / 2
        pulse = spin_system.pulse(channel.name, phase=phase, angle=np.pi / 2)

        # Detection operator
        detect = spin_system.Ix(nuc) - 1j * spin_system.Iy(nuc)

        # Initialise density operator
        rho = spin_system.equilibrium_operator

        # Initialise FID array
        fid = np.zeros(points, dtype="complex")

        # --- Apply π/2 pulse ---
        rho = rho.propagate(pulse)

        # --- Detection ---
        for i in range(pts):
            fid[i] = rho.expectation(detect)
            rho = rho.propagate(evol)

        fid *= np.exp(np.linspace(0, -10, points))
        return fid

    def _fid(self) -> Tuple[np.ndarray, np.ndarray]:
        pts, sw = self.points[0], self.sweep_widths[0]
        tp = np.linspace(0, (pts - 1) / sw, pts)
        if isinstance(self.fid, np.ndarray):
            return tp, self.fid

    def _spectrum(self, zf_factor: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        sw, off, pts = self.sweep_widths[0], self.offsets[0], self.points[0]
        shifts = np.linspace((sw / 2) + off, -(sw / 2) + off, pts * zf_factor)
        spectrum = fft.fftshift(
            fft.fft(
                self.fid,
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
    channel = ["1H"]
    sweep_width = ["10ppm"]
    points = [8192]
    offset = ["5ppm"]

    # Simulate the experiment
    result = pa(spin_system, points, sweep_width, offset, channel)

    # Extract FID and timepoints
    tp, fid = result.fid()
    # Extract spectrum and chemical shifts
    shifts, spectrum = result.spectrum(zf_factor=4)

    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(tp, np.real(fid))
    axs[1].plot(shifts, np.real(spectrum))
    axs[1].set_xlim(reversed(axs[1].get_xlim()))
    plt.show()
