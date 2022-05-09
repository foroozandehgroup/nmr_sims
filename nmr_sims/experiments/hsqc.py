# hsqc.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 09 May 2022 12:41:06 BST

"""Module for simulating HSQC experiments.

**Pulse Sequence:**

.. image:: ../figures/hsqc/hsqc.png

The result of this pulse sequence is a pair of amplitude-modulated FIDs.
"""

import copy
from typing import Optional, Tuple, Union

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
        decouple_f2: bool = True,
    ) -> None:
        super().__init__(spin_system, points, sweep_widths, offsets, channels)
        self.name = f"{self.channels[1].ssname}-{self.channels[0].ssname} HSQC"
        self.tau = tau
        self.decouple_f2 = decouple_f2

    def _pulse_sequence(self) -> np.ndarray:
        nuc1, nuc2 = [channel.name for channel in self.channels]
        off1, off2 = self.offsets
        sw1, sw2 = self.sweep_widths
        pts1, pts2 = self.points

        # Hamiltonian
        hamiltonian = self.spin_system.hamiltonian(offsets={nuc1: off1, nuc2: off2})

        # Evolution operator for t2
        if self.decouple_f2:
            hamiltonian_decoup = self.spin_system.hamiltonian(
                offsets={nuc1: off1, nuc2: off2},
                decouple=nuc1,
            )
            evol2 = hamiltonian_decoup.rotation_operator(1 / sw2)
        else:
            evol2 = hamiltonian.rotation_operator(1 / sw2)

        # Detection operator
        detect = self.spin_system.Ix(nuc2) + 1j * self.spin_system.Iy(nuc2)

        # Itialise FID object
        fid = {
            "cos": np.zeros((pts1, pts2), dtype="complex"),
            "sin": np.zeros((pts1, pts2), dtype="complex"),
        }

        # Initialise denistiy matrix
        rho = self.spin_system.equilibrium_operator

        # --- INEPT block ---
        evol1_inept = hamiltonian.rotation_operator(tau)
        # Inital π/2 pulse
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
            evol1_t1 = hamiltonian.rotation_operator(0.5 * i / sw1)
            # First half of t1 evolution
            rho_t1 = rho_t1.propagate(evol1_t1)
            # π pulse
            rho_t1 = rho_t1.propagate(self.pulses[2]["x"]["180"])
            # Second half of t1 evolution
            rho_t1 = rho_t1.propagate(evol1_t1)

            # --- Reverse INEPT block ---
            rho_t1 = rho_t1.propagate(self.pulses[2]["x"]["90"])
            for phase, comp in zip(("x", "y"), ("cos", "sin")):
                rho_t1_phase = copy.deepcopy(rho_t1)
                rho_t1_phase = rho_t1_phase.propagate(self.pulses[1][phase]["90"])
                # First half of reverse INEPT evolution
                rho_t1_phase = rho_t1_phase.propagate(evol1_inept)
                # Inversion pulses
                rho_t1_phase = rho_t1_phase.propagate(self.pulses[1]["x"]["180"])
                rho_t1_phase = rho_t1_phase.propagate(self.pulses[2]["x"]["180"])
                # Second half of reverse INEPT evolution
                rho_t1_phase = rho_t1_phase.propagate(evol1_inept)

                # --- Detection ---
                for j in range(pts2):
                    fid[comp][i, j] = rho_t1_phase.expectation(detect)
                    rho_t1_phase = rho_t1_phase.propagate(evol2)

        return fid

    def fid(
        self,
        lb: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, Tuple[str, str]]:
        """Return the FID associated with a simulation.

        Parameters
        ----------
        lb
            Line-broadening factor for exponential window function. Default
            option (``None``), will apply an exponential window such that the final
            point in each dimension will be shrunk to 1/1000 of its original value.

        Returns
        -------
        timepoints
            The timepoints sampled.

        fid
            The FID sampled.

        labels
            Axis labels for plotting purposes.
        """
        pts1, pts2 = self.points
        sw1, sw2 = self.sweep_widths
        tp = np.meshgrid(
            np.linspace(0, (pts2 - 1) / sw2, pts2),
            np.linspace(0, (pts1 - 1) / sw1, pts1),
            indexing="ij",
        )

        if lb is None:
            lb = [-np.log(0.001) / (np.pi * (p - 1)) for p in self.points]
            print(lb)
        em = np.einsum(
            "i,j->ij",
            np.exp(-np.pi * np.arange(pts1) * lb[0]),
            np.exp(-np.pi * np.arange(pts2) * lb[1]),
        )

        fid = {
            "cos": self._fid["cos"] * em,
            "sin": self._fid["sin"] * em,
        }

        return tp, fid, ("$t_1$ (s)", "$t_2$ (s)")

    def spectrum(
        self,
        zf_factor: Tuple[float, float] = [1.0, 1.0],
        lb: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, Tuple[float, float]]:
        """Return the spectrum associated with a simulation.

        Parameters
        ----------
        zf_factor
            The ratio between the number of points in the final spectrum,
            generated by zero-filling the FID, and the FID itself in each dimension.
            ``[1.0, 1.0]`` (default) means no zero-filling is applied. Each value
            should be ``>= 1.0``.

        lb
            Line-broadening factor for exponential window function. Default
            option (``None``), will apply an exponential window such that the final
            point in each dimension will be shrunk to 1/1000 of its original value.

        Returns
        -------
        shifts
            The chemical shifts sampled.

        spectrum
            The spectrum generated from the FID.

        labels
            Axis labels for plotting purposes.
        """
        off1, off2 = self.offsets
        pts1, pts2 = self.points
        sfo1, sfo2 = self.sfo
        sw1, sw2 = self.sweep_widths
        shape1, shape2 = [int(pts * zf) for pts, zf in zip(self.points, zf_factor)]

        shifts = np.meshgrid(
            np.linspace((sw1 / 2) + off1, -(sw1 / 2) + off1, shape1) / sfo1,
            np.linspace((sw2 / 2) + off2, -(sw2 / 2) + off2, shape2) / sfo2,
            indexing="ij",
        )

        _, fid, _ = self.fid(lb=lb)
        cos = fid["cos"]
        cos[0, 0] /= 2
        sin = fid["sin"]
        sin[0, 0] /= 2

        cos_f2 = np.real(fft.fftshift(fft.fft(cos, shape2, axis=1), axes=1))
        sin_f2 = np.real(fft.fftshift(fft.fft(sin, shape2, axis=1), axes=1))
        spectrum = fft.fftshift(
            fft.fft(
                cos_f2 + 1j * sin_f2,
                shape1,
                axis=0,
            ),
            axes=0,
        )

        labels = tuple([f"{channel.ssname} (ppm)" for channel in self.channels])

        return shifts, spectrum, labels


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("tkAgg")

    points = [128, 512]
    sw = ["20ppm", "5ppm"]
    off = ["50ppm", "2ppm"]
    nuc = ["13C", "1H"]

    ss = SpinSystem(
        {
            1: {
                "shift": 3,
                "couplings": {
                    2: 40,
                },
            },
            2: {
                "nucleus": "13C",
                "shift": 51,
            },
            3: {
                "shift": 0.5,
                "couplings": {
                    4: 40,
                },
            },
            4: {
                "nucleus": "13C",
                "shift": 44,
            },
        }
    )
    tau = 1 / (4 * 40)
    hsqc = HSQCSimulation(ss, points, sw, off, nuc, tau)
    hsqc.simulate()
    shifts, spectrum, labels = hsqc.spectrum(zf_factor=[4.0, 4.0], lb=[0.02, 0.02])

    fig = plt.figure()
    ax = fig.add_subplot()

    number = 10
    base = 0.01
    factor = 1.3
    levels = [base * (factor ** i) for i in range(number)]

    ax.contour(shifts[1].T, shifts[0].T, spectrum.real.T, levels=levels, linewidths=0.6)
    ax.set_xlim(reversed(ax.get_xlim()))
    ax.set_ylim(reversed(ax.get_ylim()))
    ax.set_xlabel(labels[1])
    ax.set_ylabel(labels[0])
    plt.show()
