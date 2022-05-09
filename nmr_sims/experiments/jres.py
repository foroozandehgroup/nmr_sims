# jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 09 May 2022 11:35:41 BST

"""Module for simulating homonuclear J-Resolved (2DJ) experiments.

**Pulse Sequence:**

.. image:: ../figures/jres/jres.png
"""

from typing import Optional, Tuple, Union
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
        fid = np.zeros((pts1, pts2), dtype="complex")

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
                fid[i, j] = rho.expectation(Iminus)
                rho = rho.propagate(evol2)

        return fid

    def fid(
        self,
        lb: Tuple[float, float] = [0.0, 0.0],
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[str, str]]:
        """Return the FID associated with a simulation.

        Parameters
        ----------
        lb
            Line-broadening factor for exponential window function in each dimension.
            Default option (``[0.0, 0.0]``), will lead to no damping being be applied
            to the FID (akin to a situation where no relaxation has taken place).

        Returns
        -------
        timepoints
            The timepoints sampled.

        fid
            The FID sampled.

        labels
            Axis labels for plotting purposes.
        """
        self._check_if_fid_is_none()
        pts1, pts2 = self.points
        sw1, sw2 = self.sweep_widths
        tp = np.meshgrid(
            np.linspace(0, (pts1 - 1) / sw1, pts1),
            np.linspace(0, (pts2 - 1) / sw2, pts2),
            indexing="ij",
        )
        em = np.einsum(
            "i,j->ij",
            np.exp(-np.pi * np.arange(pts1) * lb[0]),
            np.exp(-np.pi * np.arange(pts2) * lb[1]),
        )

        return tp, self._fid * em, ("$t_1$ (s)", "$t_2$ (s)")

    def spectrum(
        self,
        zf_factor: Tuple[float, float] = [1.0, 1.0],
        lb: Tuple[float, float] = [5.0, 5.0],
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """Return the spectrum associated with a simulation.

        Parameters
        ----------
        zf_factor
            The ratio between the number of points in the final spectrum,
            generated by zero-filling the FID, and the FID itself in each dimension.
            ``[1.0, 1.0]`` (default) means no zero-filling is applied. Each value
            should be ``>= 1.0``.

        lb
            Line-broadening factor for exponential window function in each dimension.

        Returns
        -------
        shifts
            The chemical shifts sampled.

        spectrum
            The spectrum generated from the FID.

        labels
            Axis labels for plotting purposes.
        """
        shape = [int(pts * zf) for pts, zf in zip(self.points, zf_factor)]
        shifts = np.meshgrid(
            np.linspace(
                (self.sweep_widths[0] / 2),
                -(self.sweep_widths[0] / 2),
                shape[0],
            ),
            np.linspace(
                ((self.sweep_widths[1] / 2) + self.offsets[0]) / self.sfo[0],
                ((-self.sweep_widths[1] / 2) + self.offsets[0]) / self.sfo[0],
                shape[1],
            ),
            indexing="ij",
        )
        _, fid, _ = self.fid(lb=lb)
        import matplotlib as mpl
        mpl.use("tkAgg")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x, y = np.meshgrid(*[np.arange(s) for s in fid.shape], indexing="ij")
        ax.plot_wireframe(x, y, fid)
        plt.show()
        fid[0, 0] /= 2
        spectrum = np.abs(
            np.flip(
                fft.fftshift(
                    fft.fft(
                        fft.fft(
                            fid,
                            shape[0],
                            axis=0,
                        ),
                        shape[1],
                        axis=1,
                    )
                )
            )
        )

        return shifts, spectrum, ("Hz", f"{self.channels[0].ssname} (ppm)")


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("tkAgg")

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
    # Extract spectrum and chemical shifts
    shifts, spectrum, labels = sim.spectrum(zf_factor=[4.0, 4.0], lb=[0.03, 0.01])

    nlevels = 10
    baselev = 0.02
    factor = 1.3
    levels = [baselev * (factor ** i) for i in range(nlevels)]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contour(shifts[1].T, shifts[0].T, spectrum.T, levels=levels, linewidths=0.6)
    ax.set_xlim(reversed(ax.get_xlim()))
    ax.set_ylim(reversed(ax.get_ylim()))
    ax.set_xlabel(labels[1])
    ax.set_ylabel(labels[0])
    plt.show()
