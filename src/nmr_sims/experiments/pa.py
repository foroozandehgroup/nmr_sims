# pa.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sat 15 Jan 2022 16:45:35 GMT

import numpy as np
from numpy import fft
from nmr_sims import _sanity
from nmr_sims.experimental import Parameters
from nmr_sims.spin_system import SpinSystem
from nmr_sims.experiments import Result, SAMPLE_SPIN_SYSTEM


def pa(spin_system: SpinSystem, parameters: Parameters) -> np.ndarray:
    nucleus, sw, offset, points = _process_params(parameters, spin_system.field)

    nuc_name = u''.join(
        dict(zip(u"0123456789", u"⁰¹²³⁴⁵⁶⁷⁸⁹")).get(c, c) for c in nucleus.name
    )

    print(
        f"Simulating {nuc_name} pulse-acquire experiment.\n"
        f"Temperature: {spin_system.temperature}K\n"
        f"Field Strength: {spin_system.field}T\n"
        f"Sweep width: {sw}Hz\n"
        f"Transmitter offset: {offset}Hz\n"
        f"Points sampled: {points}\n"
    )

    # Hamiltonian propagator
    hamiltonian = spin_system.hamiltonian(offsets={nucleus.name: offset})
    evol = hamiltonian.rotation_operator(1 / sw)

    # pi / 2 pulse propagator
    pulse = spin_system.pulse(nucleus.name, phase=np.pi / 2, angle=np.pi / 2)

    # Detection operator
    Iminus = spin_system.Ix(nucleus.name) - 1j * spin_system.Iy(nucleus.name)

    # Set density operator to be in equilibrium state
    rho = spin_system.equilibrium_operator

    fid = np.zeros(points, dtype="complex")

    # --- Run the experiment ---
    # Apply pulse
    rho = rho.propagate(pulse)
    for i in range(points):
        fid[i] = rho.expectation(Iminus)
        rho = rho.propagate(evol)

    fid *= np.exp(np.linspace(0, -10, points))

    dim_info = [{"nuc": nucleus, "sw": sw, "off": offset, "pts": points}]
    return PulseAcquireResult({"fid": fid}, dim_info, spin_system.field)


def _process_params(params: Parameters, field: float):
    nucleus = _sanity.process_nucleus(params.channels[0], None)
    sw = _sanity.process_sweep_width(params.sweep_widths[0], nucleus, field)
    offset = _sanity.process_offset(params.offsets[0], nucleus, field)
    points = params.points[0]
    return nucleus, sw, offset, points


class PulseAcquireResult(Result):
    def __init__(self, fid, dim_info, field):
        super().__init__(fid, dim_info, field)

    def fid(self):
        tp = np.linspace(0, (self.pts[0] - 1) / self.sw()[0], self.pts[0])
        fid = self._fid["fid"]
        return tp, fid

    def spectrum(self, zf_factor: int = 1):
        sw, off, pts = self.sw(unit="ppm")[0], self.offset(unit="ppm")[0], self.pts[0]
        shifts = np.linspace((sw / 2) + off, -(sw / 2) + off, pts * zf_factor)
        spectrum = fft.fftshift(
            fft.fft(
                self._fid["fid"],
                pts * zf_factor,
            )
        )

        return shifts, np.flip(spectrum)


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
        sweep_widths=["10ppm"],
        points=[8192],
        offsets=["5ppm"],
    )

    # Simulate the experiment
    result = pa(spin_system, params)
    # Extract FID and timepoints
    tp, fid = result.fid()
    # Extract spectrum and chemical shifts
    shifts, spectrum = result.spectrum(zf_factor=4)

    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(tp, np.real(fid))
    axs[1].plot(shifts, np.real(spectrum))
    axs[1].set_xlim(reversed(axs[1].get_xlim()))
    plt.show()
