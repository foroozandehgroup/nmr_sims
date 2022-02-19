# hsqc.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 18 Feb 2022 15:52:29 GMT

"""Module for simulating HSQC experiments."""

import copy
from typing import Tuple, Union

import numpy as np
from numpy import fft

from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem
from nmr_sims.experiments import Result, Simulation


class HSQC(Simulation):
    dimension_number = 2
    dimension_labels = ["1", "2"]
    channel_number = 2
    name = "HSQC"
    channel_mapping = [0, 1]

    def __init__(
        self,
        spin_system: SpinSystem,
        points: Tuple[int],
        sweep_widths: Tuple[Union[str, float, int]],
        offsets: Tuple[Union[str, float, int]],
        channels: Tuple[Union[str, Nucleus]],
        tau: float,
    ) -> None:
        super().__init__(spin_system, points, sweep_widths, offsets, channels)

    def pulse_sequence(self) -> np.ndarray:
        nuc1, nuc2 = [channel.name for channel in self.channels]
        off1, off2 = self.offsets
        sw1, sw2 = self.sweep_widths
        pts1, pts2 = self.points

        phase1 = np.pi / 2
        phase2 = phase1 + (np.pi / 2)

        # Channel 1 pulses
        pi_over_2_x_1 = self.spin_system.pulse(nuc1, phase=phase1, angle=np.pi / 2)
        pi_over_2_y_1 = self.spin_system.pulse(nuc1, phase=phase2, angle=np.pi / 2)
        pi_x_1 = self.spin_system.pulse(nuc1, phase=phase1, angle=np.pi)

        # Channel 2 pulses
        pi_over_2_x_2 = self.spin_system.pulse(nuc2, phase=phase1, angle=np.pi / 2)
        pi_x_2 = self.spin_system.pulse(nuc2, phase=phase1, angle=np.pi)

        # Hamiltonian
        hamiltonian = self.spin_system.hamiltonian(offsets={nuc1: off1, nuc2: off2})

        # Evolution operator for t2
        evol2 = hamiltonian.rotation_operator(1 / sw2)

        # Detection operator
        detect = self.spin_system.Ix(nuc1) - 1j * self.spin_system.Iy(nuc1)

        # Itialise FID object
        fid = np.zeros((pts1, pts2), dtype="complex")

        # Initialise denistiy matrix
        rho = self.spin_system.equilibrium_operator

        # --- INEPT block ---
        evol1_inept = hamiltonian.rotation_operator(tau)
        # Inital channel 1 π/2 pulse
        rho = rho.propagate(pi_over_2_x_1)
        # First half of INEPT evolution
        rho = rho.propagate(evol1_inept)
        # Inversion pulses
        rho = rho.propagate(pi_x_1).propagate(pi_x_2)
        # Second half of INEPT evolution
        rho = rho.propagate(evol1_inept)
        # Transfer onto indirect spins
        rho = rho.propagate(pi_over_2_y_1).propagate(pi_over_2_x_2)

        for i in range(pts1):
            # --- t1 evolution block ---
            rho_t1 = copy.deepcopy(rho)
            evol1_t1 = hamiltonian.rotation_operator(i / (2 * sw1))
            # First half of t1 evolution
            rho_t1 = rho_t1.propagate(evol1_t1)
            # pi pulse on channel 1
            rho_t1 = rho_t1.propagate(pi_x_1)
            # Second half of t1 evolution
            rho_t1 = rho_t1.propagate(evol1_t1)

            # --- Reverse INEPT block ---
            rho_t1 = rho_t1.propagate(pi_over_2_x_1).propagate(pi_over_2_x_2)
            # First half of reverse INEPT evolution
            rho_t1 = rho_t1.propagate(evol1_inept)
            # Inversion pulses
            rho_t1 = rho_t1.propagate(pi_x_1).propagate(pi_x_2)
            # Second half of reverse INEPT evolution
            rho_t1 = rho_t1.propagate(evol1_inept)

            # --- Detection ---
            for j in range(pts2):
                fid[i, j] = rho_t1.expectation(detect)
                rho_t1 = rho_t1.propagate(evol2)

        fid *= np.outer(
            np.exp(np.linspace(0, -5, points[0])),
            np.exp(np.linspace(0, -5, points[1])),
        )

        return fid

        spectrum = np.abs(
            np.flip(
                fft.fftshift(
                    fft.fft(
                        fft.fft(
                            fid,
                            axis=0,
                        ),
                        axis=1,
                    )
                )
            )
        )
        shifts = np.meshgrid(
            np.linspace((sw1 / 2) + off1, -(sw1 / 2) + off1, spectrum.shape[0]),
            np.linspace((sw2 / 2) + off2, -(sw2 / 2) + off2, spectrum.shape[1]),
            indexing="ij",
        )
        import matplotlib
        matplotlib.use("tkAgg")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot_wireframe(*shifts, spectrum)
        plt.show()


if __name__ == "__main__":
    points = [128, 2024]
    sw = ["10ppm", "5ppm"]
    off = ["50ppm", "2ppm"]
    nuc = ["13C", "1H"]

    ss = SpinSystem(
        {
            1: {
                "shift": 4,
                "couplings": {
                    2: 92,
                },
            },
            2: {
                "shift": 54,
                "nucleus": "13C",
            }
        }
    )
    tau = 1 / (4 * 92)

    hsqc = HSQC(ss, points, sw, off, nuc, tau)
    hsqc.simulate()
