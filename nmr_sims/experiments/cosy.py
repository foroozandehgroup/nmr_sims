# cosy.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 31 Dec 2021 13:50:08 GMT

import copy
from typing import Dict
import numpy as np
from numpy import fft
from nmr_sims import Operator
from nmr_sims.experiments import get_spin_system


def cosy(
    spin_system: Dict[int, Dict[str, float]],
    sw: float,
    tp: int = 512,
):

    spin_system = get_spin_system(spin_system)
    zero_op = Operator(0.5, spin_system.nspins)
    indices = range(1, spin_system.nspins + 1)
    Ix = sum(
        [spin_system.get(f"{i}x") for i in indices],
        start=zero_op
    )
    Iy = sum(
        [spin_system.get(f"{i}y") for i in indices],
        start=zero_op
    )
    Iz = sum(
        [spin_system.get(f"{i}z") for i in indices],
        start=zero_op
    )
    Iminus = Ix - 1j * Iy

    H = spin_system.free_hamiltonian
    dt = 1 / sw
    evol = H.rotation_operator(dt)
    pulse_1s = [-Iy.rotation_operator(np.pi / 2), -Ix.rotation_operator(np.pi / 2)]
    pulse_2 = Ix.rotation_operator(np.pi / 2)
    fids = [np.zeros((tp, tp), dtype="complex"), np.zeros((tp, tp), dtype="complex")]
    for fid, pulse_1 in zip(fids, pulse_1s):
        rho_t1 = Iz
        rho_t1 = rho_t1.propagate(pulse_1)
        for i in range(tp):
            rho_t2 = copy.deepcopy(rho_t1)
            rho_t2 = rho_t2.propagate(pulse_2)
            for j in range(tp):
                fid[i, j] = rho_t2.expectation(Iminus)
                rho_t2 = rho_t2.propagate(evol)
            rho_t1 = rho_t1.propagate(evol)

    window = np.outer(
        np.exp(np.linspace(0, -0.5, tp)),
        np.exp(np.linspace(0, -0.5, tp)),
    )
    fids[0] *= window
    fids[1] *= window

    s_t1_omega2 = (
        np.real(fft.fftshift(fft.fft(fids[0], axis=1), axes=1)) +
        1j * np.real(fft.fftshift(fft.fft(fids[1], axis=1), axes=1))
    )
    spectrum = np.real(fft.fftshift(fft.fft(s_t1_omega2, axis=0), axes=0))

    shifts = np.linspace(-2500, 2500, tp)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("tkAgg")
    fig = plt.figure()
    ax = fig.add_subplot()
    x, y = np.meshgrid(
        shifts, shifts, indexing="ij"
    )
    levels = [100, 200, 400, 800, 1600]
    ax.contour(x, y, spectrum, levels=levels)
    plt.show()


if __name__ == "__main__":
    ss = {
        1: {
            "shift": 500,
            "couplings": {
                2: 40,
                3: 40,
            }
        },
        2: {
            "shift": -1000,
            "couplings": {
                1: 40,
                3: 40,
            }
        },
        3: {
            "shift": 1500,
            "couplings": {
                1: 40,
                2: 40,
            }
        },
    }
    cosy(ss, 5000)
