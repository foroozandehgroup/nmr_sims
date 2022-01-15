# pa.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sat 15 Jan 2022 16:45:35 GMT

import numpy as np
from numpy import fft
from nmr_sims.experimental import Experimental
from nmr_sims.spin_system import SpinSystem


def pa(spin_system: SpinSystem, experimental: Experimental) -> np.ndarray:
    spin_system.set_conditions(experimental)
    channel = experimental.channels[1]

    # Hamiltonian for the system
    hamiltonian = spin_system.hamiltonian

    # Hamiltonian propagator
    dt = 1 / channel.sweep_width
    evol = hamiltonian.rotation_operator(dt)

    # Detection operator
    Iminus = spin_system.Ix - 1j * spin_system.Iy

    # pi / 2 pulse propagator
    pulse = spin_system.pulse(phase=np.pi / 2, angle=np.pi / 2)

    # Set density operator to be in equilibrium state
    rho = spin_system.equilibrium_operator

    fid = np.zeros(channel.points, dtype="complex")

    # --- Run the experiment ---
    # Apply pulse
    rho = rho.propagate(pulse)
    for i in range(channel.points):
        fid[i] = rho.expectation(Iminus)
        rho = rho.propagate(evol)

    return fid


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("tkAgg")

    # Efectively an AX3 spin system.
    spin_system = SpinSystem(
        {
            1: {
                "shift": 2,
                "couplings": {
                    2: 40.,
                    3: 40.,
                    4: 40.,
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
            "sweep_width": "10ppm",
            "offset": "5ppm",
            "points": 8192
        },
    }
    experimental = Experimental(channel, "298K", "500MHz")

    fid = pa(spin_system, experimental)

    channel = experimental.channels[1]
    window = np.exp(np.linspace(0, 8, channel.points))
    fid /= window
    spectrum = fft.fftshift(
        fft.fft(
            fid,
            4 * fid.size,
        ),
    )

    shifts = np.linspace(
        -channel.sweep_width / 2 + channel.offset,
        channel.sweep_width / 2 + channel.offset,
        spectrum.size
    ) / (1e-6 * channel.nucleus.gamma * experimental.field / (2 * np.pi))

    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(fid)
    axs[1].plot(shifts, spectrum)
    axs[1].set_xlim(reversed(axs[1].get_xlim()))
    plt.show()
