# jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sat 15 Jan 2022 19:43:46 GMT

import copy
import numpy as np
from numpy import fft
from nmr_sims.experimental import Experimental
from nmr_sims.spin_system import SpinSystem


def jres(spin_system: SpinSystem, experimental: Experimental) -> np.ndarray:
    spin_system.set_conditions(experimental)
    channel1 = experimental.channels[1]
    channel2 = experimental.channels[2]

    # Hamiltonian for the system
    hamiltonian = spin_system.hamiltonian

    # Hamiltonian propagators
    evol2 = hamiltonian.rotation_operator(1 / channel2.sweep_width)

    # Pulses
    pi_over_2_x = spin_system.pulse(phase=0, angle=np.pi / 2)
    pi_x = spin_system.pulse(phase=0, angle=np.pi)

    # Detection operator
    Iminus = spin_system.Ix - 1j * spin_system.Iy

    tp1 = channel1.points
    tp2 = channel2.points
    fid = np.zeros((tp1, tp2), dtype="complex")

    for i in range(tp1):
        # Set density matrix to Equilibrium operator
        rho_t1 = spin_system.equilibrium_operator
        # π/2 x-pulse
        rho_t1 = rho_t1.propagate(pi_over_2_x)
        evol1 = hamiltonian.rotation_operator(i / (2 * channel1.sweep_width))
        # Free evolution (first half of t1)
        rho_t1 = rho_t1.propagate(evol1)
        # π x-pulse
        rho_t1 = rho_t1.propagate(pi_x)
        # Free evolution (second half of t1)
        rho_t1 = rho_t1.propagate(evol1)
        rho_t2 = copy.deepcopy(rho_t1)
        for j in range(tp2):
            fid[i, j] = rho_t2.expectation(Iminus)
            rho_t2 = rho_t2.propagate(evol2)

    return fid


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("tkAgg")

    spin_system = SpinSystem(
        {
            1: {
                "shift": 2,
                "couplings": {
                    2: 30.,
                    3: 30.,
                    4: 30.,
                }
            },
            2: {
                "shift": 8,
            },
            3: {
                "shift": 8,
            },
            4: {
                "shift": 8,
            },
        }
    )

    channel = {
        1: {
            "nucleus": "1H",
            "sweep_width": "100Hz",
            "offset": 0,
            "points": 64
        },
        2: {
            "nucleus": "1H",
            "sweep_width": "10ppm",
            "offset": "5ppm",
            "points": 256
        },
    }
    experimental = Experimental(channel, "298K", "500MHz")
    channel1 = experimental.channels[1]
    channel2 = experimental.channels[2]

    fid = jres(spin_system, experimental)
    tp1 = channel1.points
    tp2 = channel2.points

    window = np.exp(
        np.outer(
            1 * np.linspace(0, 1, tp1),
            5 * np.linspace(0, 1, tp2),
        )
    )
    spectrum = np.abs(
        fft.fftshift(
            fft.fft(
                fft.fft(
                    fid * window,
                    4 * tp1,
                    axis=0
                ),
                4 * tp2,
                axis=1,
            ),
        )
    )
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x, y = np.meshgrid(
        np.arange(spectrum.shape[0]), np.arange(spectrum.shape[1]), indexing="ij"
    )
    ax.plot_surface(x, y, spectrum, rstride=2, cstride=2)
    plt.show()
