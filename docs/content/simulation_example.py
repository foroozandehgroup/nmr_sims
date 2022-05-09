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
# Zero fill to 16k points by setting zf_factor to 2
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
# Zero pad both dimensions by a factor of 4
shifts, spectrum, labels = jres_simulation.spectrum(zf_factor=[4.0, 4.0])

# Contour level specification
nlevels = 6
base = 0.015
factor = 1.4
levels = [base * (factor ** i) for i in range(nlevels)]

# Create the figure
# N.B. spectrum() returns the data such that axis 0 (rows) correspond to F1 and
# axis 1 (columns) correspond to F2. By convention in NMR, the direct dimension
# is typically plotted on the x-axis in figures, so we need to have shifts[1] as
# x and shifts[0] as y. Also, everything has to be transposed.
fig, ax = plt.subplots()
ax.contour(shifts[1].T, shifts[0].T, spectrum.real.T, levels=levels)
ax.set_xlabel(labels[1])
ax.set_ylabel(labels[0])
ax.set_xlim(reversed(ax.get_xlim()))
ax.set_ylim(reversed(ax.get_ylim()))
fig.savefig("jres_spectrum.png")
