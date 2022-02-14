# pa.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 14 Feb 2022 17:15:47 GMT

from typing import Tuple, Union

import numpy as np
from numpy import fft

from nmr_sims.experiments import _process_params, Result, SAMPLE_SPIN_SYSTEM
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem


def pa(
    spin_system: SpinSystem,
    points: Tuple[int],
    sweep_width: Tuple[Union[str, float, int]],
    offset: Tuple[Union[str, float, int]] = [0.0],
    channel: Tuple[Union[str, Nucleus]] = ["1H"],
) -> np.ndarray:
    points, sweep_width, offset, channel = _process_params(
        1, 1, [0], points, sweep_width, offset, channel, spin_system.field,
    )
    points, sweep_width, offset, channel = (
        points[0], sweep_width[0], offset[0], channel[0],
    )
    channel_name = u''.join(
        dict(zip(u"0123456789", u"⁰¹²³⁴⁵⁶⁷⁸⁹")).get(c, c) for c in channel.name
    )

    print(
        f"Simulating {channel_name} pulse-acquire experiment.\n"
        f"Temperature: {spin_system.temperature}K\n"
        f"Field Strength: {spin_system.field}T\n"
        f"Sweep width: {sweep_width}Hz\n"
        f"Transmitter offset: {offset}Hz\n"
        f"Points sampled: {points}\n"
    )

    # Hamiltonian propagator
    hamiltonian = spin_system.hamiltonian(offsets={channel.name: offset})
    evol = hamiltonian.rotation_operator(1 / sweep_width)

    # pi / 2 pulse propagator
    pulse = spin_system.pulse(channel.name, phase=np.pi / 2, angle=np.pi / 2)

    # Detection operator
    Iminus = spin_system.Ix(channel.name) - 1j * spin_system.Iy(channel.name)

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

    dim_info = [{"nuc": channel, "sw": sweep_width, "off": offset, "pts": points}]
    return PulseAcquireResult({"fid": fid}, dim_info, spin_system)


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
    channel = ["1H"]
    sweep_width = ["10ppm"]
    points = [8192]
    offset = ["5ppm"]

    # Simulate the experiment
    result = pa(spin_system, points, sweep_width, offset, channel)

    # Extract FID and timepoints
    tp, fid = result.fid()
    # Extract spectrum and chemical shifts
    shifts, spectrum = result.spectrum(zf_factor=4)

    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(tp, np.real(fid))
    axs[1].plot(shifts, np.real(spectrum))
    axs[1].set_xlim(reversed(axs[1].get_xlim()))
    plt.show()
