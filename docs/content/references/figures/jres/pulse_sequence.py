# pulse_sequence.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 28 Feb 2022 15:47:22 GMT

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


PULSE_WIDTH = 1
PULSE_HEIGHT = 0.3  # fraction of height of figure
T1_WIDTH = 0.05
HORIZOTAL_PADS = (0.05, 0.01)

# --- horizontal dimensions---
LEFT_PAD = 2
NINETY_WIDTH = 2
T1_WIDTH = 9
T2_WIDTH = 15
RIGHT_PAD = 2

# ---vertical dimensions---
CHANNEL_HEIGHT = 8
TOP_PAD = 1
BOTTOM_PAD = 4.5


def add_pulse(ax, x0, flip_angle="90"):
    y0 = BOTTOM_PAD
    if flip_angle == "90":
        width = NINETY_WIDTH
        fc = "k"
    elif flip_angle == "180":
        width = 2 * NINETY_WIDTH
        fc = "w"

    ax.add_patch(
        Rectangle(
            (x0, y0), width, CHANNEL_HEIGHT, facecolor=fc, edgecolor="k",
            transform=ax.transAxes, linewidth=1,
        ),
    )


def add_text(ax, x, y, txt):
    ax.text(
        x, y, txt, horizontalalignment="center", verticalalignment="center",
        transform=ax.transAxes, fontsize=10,
    )


# ==============================
horizontal_total = (
    LEFT_PAD + RIGHT_PAD + 2 * T1_WIDTH + 3 * NINETY_WIDTH + T2_WIDTH
)
LEFT_PAD, RIGHT_PAD, T1_WIDTH, NINETY_WIDTH, T2_WIDTH = [
    x / horizontal_total for x in
    [LEFT_PAD, RIGHT_PAD, T1_WIDTH, NINETY_WIDTH, T2_WIDTH]
]

vertical_total = BOTTOM_PAD + TOP_PAD + CHANNEL_HEIGHT
BOTTOM_PAD, TOP_PAD, CHANNEL_HEIGHT = [
    x / vertical_total for x in
    [BOTTOM_PAD, TOP_PAD, CHANNEL_HEIGHT]
]

fig = plt.figure(figsize=(3, 1))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")

ax.plot(
    [LEFT_PAD, 1 - RIGHT_PAD], [BOTTOM_PAD, BOTTOM_PAD],
    color="k", solid_capstyle="round", lw=1,
    transform=ax.transAxes,

)

# useful horizonal positions
acquisition = LEFT_PAD + 3 * NINETY_WIDTH + 2 * T1_WIDTH
add_pulse(ax, LEFT_PAD)
add_pulse(ax, LEFT_PAD + NINETY_WIDTH + T1_WIDTH, "180")

add_text(ax, LEFT_PAD + NINETY_WIDTH + 0.5 * T1_WIDTH, BOTTOM_PAD + 0.5 * CHANNEL_HEIGHT, r"$\frac{t_1}{2}$")
add_text(ax, LEFT_PAD + 3 * NINETY_WIDTH + 1.5 * T1_WIDTH, BOTTOM_PAD + 0.5 * CHANNEL_HEIGHT, r"$\frac{t_1}{2}$")

tp = np.linspace(acquisition, acquisition + T2_WIDTH, 256)
fid = BOTTOM_PAD + (
    0.5 * CHANNEL_HEIGHT * np.cos(300 * (tp - acquisition)) *
    np.exp(np.linspace(0, -4, tp.size))
)
ax.plot(tp, fid, transform=ax.transAxes, color="k", lw=1.4, solid_capstyle="round")

fig.savefig("jres.pdf")
fig.savefig("jres.png")
