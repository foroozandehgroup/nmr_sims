# cosy.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 01 Mar 2022 23:22:27 GMT

"""Module for simulating COSY experiments.

**Pulse Sequence:**

**TODO**

The result of this pulse sequence is a pair of amplitude-modulated FIDs.
"""
import copy
from typing import Dict, Union

import numpy as np
from numpy import fft

from nmr_sims.experiments import Simulation, SAMPLE_SPIN_SYSTEM
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem


class COSYSimulation(Simulation):
    dimension_number = 2
    channel_number = 1
    channel_mapping = [0, 0]

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
        super().__init__(
            spin_system, [points, points], [sweep_width, sweep_width],
            [offset], [channel],
        )
        self.name = f"{self.channels[0].ssname} COSY"

    def _pulse_sequence(self) -> Dict[str, np.ndarray]:
        pts, sw, off, nuc = (
            self.points[0], self.sweep_widths[0], self.offsets[0],
            self.channels[0].name,
        )

        # Hamiltonian propagator
        hamiltonian = self.spin_system.hamiltonian(offsets={nuc: off})
        evol_t2 = hamiltonian.rotation_operator(1 / sw)

        # Detection operator
        detect = self.spin_system.Ix(nuc) + 1j * self.spin_system.Iy(nuc)

        # Itialise FID object
        fid = {
            "cos": np.zeros((pts, pts), dtype="complex"),
            "sin": np.zeros((pts, pts), dtype="complex"),
        }

        for comp, phase in zip(("cos", "sin"), ("x", "-y")):
            # Initialise density operator
            rho = self.spin_system.equilibrium_operator

            # --- Apply π/2 pulse ---
            rho = rho.propagate(self.pulses[1][phase]["90"])

            # --- t1 evolution ---
            for i in range(pts):
                rho_t1 = copy.deepcopy(rho)
                evol_t1 = hamiltonian.rotation_operator(i / sw)
                rho_t1 = rho_t1.propagate(evol_t1)
                rho_t1 = rho_t1.propagate(self.pulses[1]["x"]["90"])

                # --- Detection ---
                for j in range(pts):
                    fid[comp][i, j] = rho_t1.expectation(detect)
                    rho_t1 = rho_t1.propagate(evol_t2)

        fid["cos"] *= np.outer(
            np.exp(np.linspace(0, -10, pts)),
            np.exp(np.linspace(0, -10, pts)),
        )
        fid["sin"] *= np.outer(
            np.exp(np.linspace(0, -10, pts)),
            np.exp(np.linspace(0, -10, pts)),
        )

        return fid

    def _fetch_fid(self):
        pts = self.points[0]
        sw = self.sweep_widths[0]
        tp = np.meshgrid(
            np.linspace(0, (pts - 1) / sw, pts),
            np.linspace(0, (pts - 1) / sw, pts),
            indexing="ij",
        )
        return tp, self._fid

    def _fetch_spectrum(self, zf_factor: int = 1):
        off = self.offsets[0]
        pts = self.points[0]
        sfo = self.sfo[0]
        sw = self.sweep_widths[0]
        channel = self.channels[0]

        shifts = np.meshgrid(
            np.linspace((sw / 2) + off, -(sw / 2) + off, pts * zf_factor) / sfo,
            np.linspace((sw / 2) + off, -(sw / 2) + off, pts * zf_factor) / sfo,
            indexing="ij",
        )

        cos = self._fid["cos"]
        cos[0, 0] /= 2
        sin = self._fid["sin"]
        sin[0, 0] /= 2

        cos_f2 = np.real(fft.fftshift(fft.fft(cos, zf_factor * pts, axis=0), axes=0))
        sin_f2 = np.real(fft.fftshift(fft.fft(sin, zf_factor * pts, axis=0), axes=0))
        spectrum = fft.fftshift(
            fft.fft(
                cos_f2 + 1j * sin_f2,
                pts * zf_factor,
                axis=1,
            ),
            axes=1,
        )

        labels = tuple(2 * [f"{channel.ssname} (ppm)"])
        return shifts, spectrum, labels


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
    points = 256
    offset = "5ppm"

    # Simulate the experiment
    sim = COSYSimulation(spin_system, points, sweep_width, offset, channel)
    sim.simulate()
    # Extract spectrum and chemical shifts
    shifts, spectrum, labels = sim.spectrum(zf_factor=4)

    nlevels = 10
    baselev = 0.003
    factor = 1.4
    levels = [baselev * (factor ** i) for i in range(nlevels)]
    levels = [-x for x in reversed(levels)] + levels
    print(levels)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contour(shifts[0], shifts[1], spectrum, levels=levels, linewidths=0.6)
    ax.set_xlim(reversed(ax.get_xlim()))
    ax.set_ylim(reversed(ax.get_ylim()))
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    plt.show()
