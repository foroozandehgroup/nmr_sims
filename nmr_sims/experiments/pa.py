# pa.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 21 Dec 2021 12:32:14 GMT

from typing import Dict
import numpy as np
from numpy import fft
from nmr_sims import Operator
from nmr_sims.experiments import get_spin_system


def pa(
    spin_system: Dict[int, Dict[str, float]],
    sw: float,
    tp: int = 2048,
):

    spin_system = get_spin_system(spin_system)
    zero_op = Operator(0.5, spin_system.nspins)
    Ix = sum(
        [spin_system.get(f"{i}x") for i in range(1, spin_system.nspins + 1)],
        start=zero_op
    )
    Iy = sum(
        [spin_system.get(f"{i}y") for i in range(1, spin_system.nspins + 1)],
        start=zero_op
    )
    Iz = sum(
        [spin_system.get(f"{i}z") for i in range(1, spin_system.nspins + 1)],
        start=zero_op
    )
    Iminus = Ix - 1j * Iy

    H = spin_system.free_hamiltonian
    dt = 1 / sw
    evol = H.rotation_operator(dt)
    pi_over_2_x = -Iy.rotation_operator(np.pi / 2)

    # Run the experiment
    fid = np.zeros(tp, dtype="complex")
    # Eqm. magnetisation
    rho = Iz
    # π/2 x-pulse
    rho = pi_over_2_x @ rho @ pi_over_2_x.adjoint
    for i in range(tp):
        # Free evolution (first half of t1)
        fid[i] = rho.expectation(Iminus)
        rho = evol @ rho @ evol.adjoint

    window = np.exp(np.linspace(0, 8, tp))
    fid /= window
    spectrum = fft.fftshift(
        fft.fft(
            fid,
            4 * tp,
        ),
    )

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("tkAgg")
    fig, axs = plt.subplots(nrows=2, ncols=1)
    print(axs)
    axs[0].plot(fid)
    axs[1].plot(spectrum)
    plt.show()


if __name__ == "__main__":
    spin_system = {
        1: {
            "shift": 1000.,
            "couplings": {
                2: 40.,
                3: 40.,
                4: 40.,
            }
        },
        2: {
            "shift": -500.,
            "couplings": {
                1: 40.
            }
        },
        3: {
            "shift": -500.,
            "couplings": {
                1: 40.
            }
        },
        4: {
            "shift": -500.,
            "couplings": {
                1: 40.
            }
        },
    }

    pa(spin_system, 5000, 2048)
