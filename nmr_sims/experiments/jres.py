# jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 28 Feb 2022 15:31:46 GMT

"""Module for simulating homonuclear J-Resolved (2DJ) experiments.

**Pulse Sequence:**

.. image:: ../figures/jres/jres.png
"""

from typing import Tuple, Union
import numpy as np
from numpy import fft
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem
from nmr_sims.experiments import SAMPLE_SPIN_SYSTEM, Simulation


class JresSimulation(Simulation):
    """Simulation class for J-Resolved (2DJ) experiment."""
    dimension_number = 2
    channel_number = 1
    channel_mapping = [0, 0]

    def __init__(
        self,
        spin_system: SpinSystem,
        points: Tuple[int, int],
        sweep_widths: Tuple[Union[str, float, int], Union[str, float, int]],
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

        sweep_widths
            The sweep width in each dimension.

        offset
            The transmitter offset.

        channel
            The nucelus targeted by the channel.
        """
        super().__init__(spin_system, points, sweep_widths, [offset], [channel])
        self.name = f"{self.channels[0].ssname} J-Resolved"

    def _pulse_sequence(self) -> np.ndarray:
        nuc = self.channels[0].name
        off = self.offsets[0]
        pts1, pts2 = self.points
        sw1, sw2 = self.sweep_widths

        # Hamiltonian
        hamiltonian = self.spin_system.hamiltonian(offsets={nuc: off})

        # Hamiltonian propagator for t2
        evol2 = hamiltonian.rotation_operator(1 / sw2)

        # Detection operator
        Iminus = self.spin_system.Ix(nuc) - 1j * self.spin_system.Iy(nuc)

        # Initialise FID array
        fid = np.zeros((pts2, pts1), dtype="complex")

        for i in range(pts1):
            # Propagator for each half of the t1 period
            evol1 = hamiltonian.rotation_operator(i / (2 * sw1))

            # Set density matrix to Equilibrium operator
            rho = self.spin_system.equilibrium_operator

            # --- Apply π/2 pulse ---
            rho = rho.propagate(self.pulses[1]["y"]["90"])

            # --- t1 Evolution ---
            # First half of t1 evolution
            rho = rho.propagate(evol1)
            # π pulse
            rho = rho.propagate(self.pulses[1]["-x"]["180"])
            # Second half of t1 evolution
            rho = rho.propagate(evol1)

            # --- Detection ---
            for j in range(pts2):
                fid[j, i] = rho.expectation(Iminus)
                rho = rho.propagate(evol2)

        fid *= np.outer(
            np.exp(np.linspace(0, -7, pts2)),
            np.exp(np.linspace(0, -7, pts1)),
        )

        return fid

    def _fetch_fid(self):
        pts1, pts2 = self.points
        sw1, sw2 = self.sweep_widths
        tp = np.meshgrid(
            np.linspace(0, (pts2 - 1) / sw2, pts2),
            np.linspace(0, (pts1 - 1) / sw1, pts1),
            indexing="ij",
        )
        return tp, self._fid

    def _fetch_spectrum(self, zf_factor: int = 1):
        off = self.offsets[0]
        pts1, pts2 = self.points
        sfo = self.sfo[0]
        sw1, sw2 = self.sweep_widths
        shifts = np.meshgrid(
            np.linspace((sw2 / 2) + off, -(sw2 / 2) + off, pts2 * zf_factor) / sfo,
            np.linspace((sw1 / 2), -(sw1 / 2), pts1 * zf_factor),
            indexing="ij",
        )
        fid = self._fid
        fid[0, 0] /= 2
        spectrum = np.abs(
            np.flip(
                fft.fftshift(
                    fft.fft(
                        fft.fft(
                            self._fid,
                            pts2 * zf_factor,
                            axis=0,
                        ),
                        pts1 * zf_factor,
                        axis=1,
                    )
                )
            )
        )

        return shifts, spectrum, (f"{self.channels[0].ssname} (ppm)", "Hz")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # AX3 1H spin system with A @ 2ppm and X @ 7ppm.
    # Field of 500MHz
    spin_system = SAMPLE_SPIN_SYSTEM

    # Experiment parameters
    channel = "1H"
    sweep_widths = ["100Hz", "10ppm"]
    points = [64, 256]
    offset = "5ppm"

    # Simulate the experiment
    sim = JresSimulation(spin_system, points, sweep_widths, offset, channel)
    sim.simulate()
    # Extract FID and timepoints
    tp, fid = sim.fid()
    # Extract spectrum and chemical shifts
    shifts, spectrum, labels = sim.spectrum(zf_factor=4)

    nlevels = 10
    baselev = 0.02
    factor = 1.4
    levels = [baselev * (factor ** i) for i in range(nlevels)]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contour(shifts[0], shifts[1], spectrum, levels=levels, linewidths=0.6)
    ax.set_xlim(reversed(ax.get_xlim()))
    ax.set_ylim(reversed(ax.get_ylim()))
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    fig.savefig("example_figures/jres.pdf")
