# jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sat 15 Jan 2022 19:43:46 GMT

import numpy as np
from numpy import fft
from nmr_sims import _sanity
from nmr_sims.experimental import Parameters
from nmr_sims.spin_system import SpinSystem
from nmr_sims.experiments import Result, SAMPLE_SPIN_SYSTEM


def jres(spin_system: SpinSystem, parameters: Parameters) -> np.ndarray:
    nucleus, sw, offset, points = _process_params(parameters, spin_system.field)

    nuc_name = u''.join(
        dict(zip(u"0123456789", u"⁰¹²³⁴⁵⁶⁷⁸⁹")).get(c, c) for c in nucleus.name
    )

    print(
        f"Simulating {nuc_name} 2DJ experiment.\n"
        f"Temperature: {spin_system.temperature}K\n"
        f"Field Strength: {spin_system.field}T\n"
        f"Sweep width: {sw[0]}Hz (F1), {sw[1]}Hz (F2)\n"
        f"Transmitter offset: {offset}Hz\n"
        f"Points sampled: {points[0]} (F1), {points[1]} (F2)\n"
    )

    # Hamiltonian for the system
    hamiltonian = spin_system.hamiltonian(offsets={nucleus.name: offset})

    # Hamiltonian propagator for t2
    evol2 = hamiltonian.rotation_operator(1 / sw[1])

    # Pulses
    pi_over_2_x = spin_system.pulse(nucleus.name, phase=0, angle=np.pi / 2)
    pi_x = spin_system.pulse(nucleus.name, phase=0, angle=np.pi)

    # Detection operator
    Iminus = spin_system.Ix(nucleus.name) - 1j * spin_system.Iy(nucleus.name)

    fid = np.zeros((points[0], points[1]), dtype="complex")

    for i in range(points[0]):
        # Propagator for each half of the t1 period
        evol1 = hamiltonian.rotation_operator(i / (2 * sw[0]))
        # Set density matrix to Equilibrium operator
        rho = spin_system.equilibrium_operator
        # π/2 x-pulse
        rho = rho.propagate(pi_over_2_x)
        # Free evolution (first half of t1)
        rho = rho.propagate(evol1)
        # π x-pulse
        rho = rho.propagate(pi_x)
        # Free evolution (second half of t1)
        rho = rho.propagate(evol1)
        for j in range(points[1]):
            fid[i, j] = rho.expectation(Iminus)
            rho = rho.propagate(evol2)

    fid *= np.outer(
        np.exp(np.linspace(0, -5, points[0])),
        np.exp(np.linspace(0, -5, points[1])),
    )

    dim_info = [
        {"nuc": nucleus, "sw": sw[0], "off": 0., "pts": points[0]},
        {"nuc": nucleus, "sw": sw[1], "off": offset, "pts": points[1]},
    ]

    return JresResult({"fid": fid}, dim_info, spin_system.field)


def _process_params(params: Parameters, field: float):
    nucleus = _sanity.process_nucleus(params.channels[0], None)
    sw = [
        _sanity.process_sweep_width(sw, nucleus, field)
        for sw in params.sweep_widths[:2]
    ]
    offset = _sanity.process_offset(params.offsets[0], nucleus, field)
    points = params.points[:2]
    return nucleus, sw, offset, points


class JresResult(Result):
    def __init__(self, fid, dim_info, field):
        super().__init__(fid, dim_info, field)

    def fid(self):
        tp = np.meshgrid(
            np.linspace(0, (self.pts[0] - 1) / self.sw()[0], self.pts[0]),
            np.linspace(0, (self.pts[1] - 1) / self.sw()[1], self.pts[1]),
            indexing="ij",
        )
        fid = self._fid["fid"]
        return tp, fid

    def spectrum(self, zf_factor: int = 1):
        sw, off, pts = self.sw(unit="ppm"), self.offset(unit="ppm")[1], self.pts
        shifts = np.meshgrid(
            np.linspace((sw[0] / 2), -(sw[0] / 2), pts[0] * zf_factor),
            np.linspace((sw[1] / 2) + off, -(sw[1] / 2) + off, pts[1] * zf_factor),
            indexing="ij",
        )
        spectrum = np.abs(
            np.flip(
                fft.fftshift(
                    fft.fft(
                        fft.fft(
                            self._fid["fid"],
                            pts[0] * zf_factor,
                            axis=0,
                        ),
                        pts[1] * zf_factor,
                        axis=1,
                    )
                )
            )
        )

        return shifts, spectrum


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("tkAgg")

    # AX3 1H spin system with A @ 2ppm and X @ 7ppm.
    # Field of 500MHz
    spin_system = SAMPLE_SPIN_SYSTEM
    # Experiment parameters
    params = Parameters(
        channels=["1H"],
        sweep_widths=["100Hz", "10ppm"],
        points=[64, 256],
        offsets=["5ppm"],
    )

    # Simulate the experiment
    result = jres(spin_system, params)
    # Extract FID and timepoints
    tp, fid = result.fid()
    # Extract spectrum and chemical shifts
    shifts, spectrum = result.spectrum(zf_factor=4)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(shifts[0], shifts[1], spectrum, rstride=2, cstride=2)
    plt.show()
