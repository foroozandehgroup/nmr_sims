# jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 20 Dec 2021 00:11:22 GMT

import copy
import numpy as np
from numpy import fft
from nmr_sims import CartesianBasis


def jres_ax(
    omega_a: float,
    omega_x: float,
    J_ax: float,
    sw1: float,
    sw2: float,
    tp1: int = 128,
    tp2: int = 4096,
):
    dt1 = 1 / sw1
    dt2 = 1 / sw2
    basis = CartesianBasis(0.5, nspins=2)

    # Define operators
    I1x = basis.get("1x")
    I2x = basis.get("2x")
    I1y = basis.get("1y")
    I2y = basis.get("2y")
    I1z = basis.get("1z")
    I2z = basis.get("2z")
    I1xI2x = basis.get("1x2x")
    I1yI2y = basis.get("1y2y")
    I1zI2z = basis.get("1z2z")
    Ix = I1x + I2x
    Iy = I1y + I2y
    Iz = I1z + I2z
    Iplus = Ix + 1j * Iy
    Iminus = Ix - 1j * Iy

    # Hamiltonian
    H = (
        2 * np.pi * omega_a * I1z +
        2 * np.pi * omega_x * I2z +
        np.pi * J_ax * (I1xI2x + I1yI2y + I1zI2z)
    )

    # Pulses
    pi_over_2_x = Ix.rotation_operator(np.pi / 2)
    pi_x = Ix.rotation_operator(np.pi)

    # Evolution operators
    evol1 = H.rotation_operator(dt1)
    evol2 = H.rotation_operator(dt2)

    # Run the experiment
    fid = np.zeros((tp1, tp2), dtype="complex")
    # Eqm. magnetisation
    rho_t1 = Iz
    for i in range(tp1):
        # π/2 x-pulse
        rho_t1 = pi_over_2_x @ rho_t1 @ pi_over_2_x.adjoint
        # Free evolution (first half of t1)
        rho_t1 = evol1 @ rho_t1 @ evol1.adjoint
        # Select -1 coherence
        rho_t1 = Iminus * rho_t1
        # π x-pulse
        rho_t1 = pi_x @ rho_t1 @ pi_x.adjoint
        # Select +1 coherence
        rho_t1 = Iplus * rho_t1
        # Free evolution (second half of t1)
        rho_t1 = evol1 @ rho_t1 @ evol1.adjoint
        rho_t2 = copy.deepcopy(rho_t1)
        for j in range(tp2):
            fid[i, j] = rho_t2.expectation(Iminus)
            rho_t2 = evol2 @ rho_t2 @ evol2.adjoint

    window = np.exp(
        np.outer(
            1 * np.linspace(0, 1, tp1),
            5 * np.linspace(0, 1, tp2),
        )
    )
    spectrum = np.abs(
        fft.fftshift(
            fft.fft(
                fft.fftshift(
                    fft.fft(
                        fid * window,
                        4 * tp1,
                        axis=0
                    ),
                    axes=0,
                ),
                4 * tp2,
                axis=1,
            ),
            axes=1,
        )
    )
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("tkAgg")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x, y = np.meshgrid(
        np.arange(spectrum.shape[0]), np.arange(spectrum.shape[1]), indexing="ij"
    )
    ax.plot_wireframe(x, y, spectrum)
    plt.show()


if __name__ == "__main__":
    jres_ax(400, 200, 40, 5000, 200)
