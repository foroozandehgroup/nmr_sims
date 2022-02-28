# simulation_example.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 28 Feb 2022 15:32:08 GMT

import matplotlib.pyplot as plt
from nmr_sims.experiments.jres import JresSimulation
from nmr_sims.experiments.pa import PulseAcquireSimulation
from nmr_sims.spin_system import SpinSystem


system = SpinSystem(
    {
        1: {
            "shift": 3.7,
            "couplings": {
                2: -10.1,
                3: 4.3,
            },
            "nucleus": "1H",
        },
        2: {
            "shift": 3.92,
            "couplings": {
                3: 11.3,
            },
            "nucleus": "1H",
        },
        3: {
            "shift": 4.5,
            "nucleus": "1H",
        },
    }
)

# Pulse-acquire
sw = "1ppm"
offset = "4.1ppm"
points = 8192
channel = "1H"

pa_simulation = PulseAcquireSimulation(system, points, sw, offset, channel)
pa_simulation.simulate()
# Zero fill to 8192 points by setting zf_factor to 2
shifts, spectrum, label = pa_simulation.spectrum(zf_factor=2)

fig, ax = plt.subplots()
ax.plot(shifts, spectrum.real, color="k")
ax.set_xlabel(label)
ax.set_xlim(reversed(ax.get_xlim()))
fig.savefig("pa_spectrum.png")

# 2DJ
sw = ["30Hz", "1ppm"]
offset = "4.1ppm"
points = [128, 512]
channel = "1H"

jres_simulation = JresSimulation(system, points, sw, offset, channel)
jres_simulation.simulate()
shifts, spectrum, labels = jres_simulation.spectrum(zf_factor=2)

nlevels = 6
base = 0.015
factor = 1.4
levels = [base * (factor ** i) for i in range(nlevels)]

fig, ax = plt.subplots()
ax.contour(*shifts, spectrum.real, levels=levels)
ax.set_xlabel(labels[0])
ax.set_ylabel(labels[1])
ax.set_xlim(reversed(ax.get_xlim()))
ax.set_ylim(reversed(ax.get_ylim()))
fig.savefig("jres_spectrum.png")
