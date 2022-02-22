# hsqc.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 22 Feb 2022 10:48:37 GMT

"""Module for simulating HSQC experiments."""

import copy
from typing import Tuple, Union

import numpy as np
from numpy import fft

from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem
from nmr_sims.experiments import Simulation


class HSQCSimulation(Simulation):
    dimension_number = 2
    channel_number = 2
    channel_mapping = [0, 1]

    def __init__(
        self,
        spin_system: SpinSystem,
        points: Tuple[int, int],
        sweep_widths: Tuple[Union[str, float, int], Union[str, float, int]],
        offsets: Tuple[Union[str, float, int], Union[str, float, int]],
        channels: Tuple[Union[str, Nucleus], Union[str, Nucleus]],
        tau: float,
    ) -> None:
        super().__init__(spin_system, points, sweep_widths, offsets, channels)
        self.name = f"{self.channels[1].ssname}-{self.channels[0].ssname} HSQC"

    def _pulse_sequence(self) -> np.ndarray:
        nuc1, nuc2 = [channel.name for channel in self.channels]
        off1, off2 = self.offsets
        sw1, sw2 = self.sweep_widths
        pts1, pts2 = self.points

        # Hamiltonian
        hamiltonian = self.spin_system.hamiltonian(offsets={nuc1: off1, nuc2: off2})

        # Evolution operator for t2
        evol2 = hamiltonian.rotation_operator(1 / sw2)

        # Detection operator
        detect = self.spin_system.Ix(nuc2) - 1j * self.spin_system.Iy(nuc2)

        # Itialise FID object
        fid = np.zeros((pts1, pts2), dtype="complex")

        # Initialise denistiy matrix
        rho = self.spin_system.equilibrium_operator

        # --- INEPT block ---
        evol1_inept = hamiltonian.rotation_operator(tau)
        # Inital channel 1 π/2 pulse
        rho = rho.propagate(self.pulses[2]["x"]["90"])
        # First half of INEPT evolution
        rho = rho.propagate(evol1_inept)
        # Inversion pulses
        rho = rho.propagate(self.pulses[1]["x"]["180"])
        rho = rho.propagate(self.pulses[2]["x"]["180"])
        # Second half of INEPT evolution
        rho = rho.propagate(evol1_inept)
        # Transfer onto indirect spins
        rho = rho.propagate(self.pulses[1]["x"]["90"])
        rho = rho.propagate(self.pulses[2]["y"]["90"])

        for i in range(pts1):
            # --- t1 evolution block ---
            rho_t1 = copy.deepcopy(rho)
            evol1_t1 = hamiltonian.rotation_operator(i / (2 * sw1))
            # First half of t1 evolution
            rho_t1 = rho_t1.propagate(evol1_t1)
            # pi pulse on channel 1
            rho_t1 = rho_t1.propagate(self.pulses[2]["y"]["180"])
            # Second half of t1 evolution
            rho_t1 = rho_t1.propagate(evol1_t1)

            # --- Reverse INEPT block ---
            rho_t1 = rho_t1.propagate(self.pulses[1]["x"]["90"])
            rho_t1 = rho_t1.propagate(self.pulses[2]["x"]["90"])
            # First half of reverse INEPT evolution
            rho_t1 = rho_t1.propagate(evol1_inept)
            # Inversion pulses
            rho_t1 = rho_t1.propagate(self.pulses[1]["x"]["180"])
            rho_t1 = rho_t1.propagate(self.pulses[2]["x"]["180"])
            # Second half of reverse INEPT evolution
            rho_t1 = rho_t1.propagate(evol1_inept)

            # --- Detection ---
            for j in range(pts2):
                fid[i, j] = rho_t1.expectation(detect)
                rho_t1 = rho_t1.propagate(evol2)

        fid *= np.outer(
            np.exp(np.linspace(0, -5, pts1)),
            np.exp(np.linspace(0, -5, pts2)),
        )

        return fid

    def _fetch_fid(self):
        pts1, pts2 = self.points
        sw1, sw2 = self.sweep_widths
        tp = np.meshgrid(
            np.linspace(0, (pts1 - 1) / sw1, pts1),
            np.linspace(0, (pts2 - 1) / sw2, pts2),
            indexing="ij",
        )
        return tp, self._fid

    def _fetch_spectrum(self, zf_factor: int = 1):
        off1, off2 = self.offsets
        pts1, pts2 = self.points
        sfo1, sfo2 = self.sfo
        sw1, sw2 = self.sweep_widths

        shifts = np.meshgrid(
            np.linspace((sw1 / 2) + off1, -(sw1 / 2) + off1, pts1 * zf_factor) / sfo1,
            np.linspace((sw2 / 2) + off2, -(sw2 / 2) + off2, pts2 * zf_factor) / sfo2,
            indexing="ij",
        )

        spectrum = np.flip(
            fft.fftshift(
                fft.fft(
                    fft.fft(
                        self._fid,
                        pts1 * zf_factor,
                        axis=0,
                    ),
                    pts2 * zf_factor,
                    axis=1,
                )
            )
        )

        return shifts, spectrum


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("tkAgg")

    points = [256, 2024]
    sw = ["20ppm", "5ppm"]
    off = ["50ppm", "2ppm"]
    nuc = ["13C", "1H"]

    ss = SpinSystem(
        {
            1: {
                "shift": 3,
                "couplings": {
                    3: 40,
                },
            },
            2: {
                "shift": 3,
                "couplings": {
                    3: 40,
                },
            },
            3: {
                "nucleus": "13C",
                "shift": 51,
            }
        }
    )
    tau = 1 / (4 * 40)

    hsqc = HSQCSimulation(ss, points, sw, off, nuc, tau)
    hsqc.simulate()
    shifts, spectrum = hsqc.spectrum()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(shifts[0], shifts[1], spectrum, rstride=2, cstride=2)
    plt.show()
