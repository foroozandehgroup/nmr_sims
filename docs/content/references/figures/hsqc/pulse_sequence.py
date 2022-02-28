# pulse_sequence.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 28 Feb 2022 11:17:58 GMT

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


PULSE_WIDTH = 1
PULSE_HEIGHT = 0.3  # fraction of height of figure
TAU_WIDTH = 0.05
HORIZOTAL_PADS = (0.05, 0.01)

# --- horizontal dimensions---
LEFT_PAD = 3
NINETY_WIDTH = 3
TAU_WIDTH = 9
HALF_T1_WIDTH = 12
T2_WIDTH = 15
RIGHT_PAD = 3

# ---vertical dimensions---
CHANNEL_HEIGHT = 8
CHANNEL_GAP = 3
TOP_PAD = 2
BOTTOM_PAD = 1


def add_pulse(ax, channel, x0, flip_angle="90"):
    if channel == "top":
        y0 = BOTTOM_PAD + CHANNEL_HEIGHT + CHANNEL_GAP
    elif channel == "bottom":
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
            transform=ax.transAxes
        ),
    )


def add_text(ax, x, y, txt):
    ax.text(
        x, y, txt, horizontalalignment="center", verticalalignment="center",
        transform=ax.transAxes, fontsize=10,
    )


def dotted_line(ax, channel, x):
    if channel == "top":
        y0 = BOTTOM_PAD + CHANNEL_HEIGHT + CHANNEL_GAP
    elif channel == "bottom":
        y0 = BOTTOM_PAD

    ax.plot(
        [x, x], [y0, y0 + CHANNEL_HEIGHT], color="k", linestyle=":", linewidth=1.5,
        transform=ax.transAxes
    )


# ==============================
horizontal_total = (
    LEFT_PAD + RIGHT_PAD + 4 * TAU_WIDTH + 2 * HALF_T1_WIDTH +
    10 * NINETY_WIDTH + T2_WIDTH
)
LEFT_PAD, RIGHT_PAD, TAU_WIDTH, HALF_T1_WIDTH, NINETY_WIDTH, T2_WIDTH = [
    x / horizontal_total for x in
    [LEFT_PAD, RIGHT_PAD, TAU_WIDTH, HALF_T1_WIDTH, NINETY_WIDTH, T2_WIDTH]
]

vertical_total = BOTTOM_PAD + TOP_PAD + CHANNEL_GAP + 2 * CHANNEL_HEIGHT
BOTTOM_PAD, TOP_PAD, CHANNEL_GAP, CHANNEL_HEIGHT = [
    x / vertical_total for x in
    [BOTTOM_PAD, TOP_PAD, CHANNEL_GAP, CHANNEL_HEIGHT]
]

fig = plt.figure(figsize=(6, 2))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")

# Lower channel line
ax.plot(
    [LEFT_PAD, 1 - RIGHT_PAD], [BOTTOM_PAD, BOTTOM_PAD],
    color="k", solid_capstyle="round",
    transform=ax.transAxes,

)

# Higher channel line
ax.plot(
    [LEFT_PAD, 1 - RIGHT_PAD], 2 * [BOTTOM_PAD + CHANNEL_HEIGHT + CHANNEL_GAP],
    color="k", solid_capstyle="round",
    transform=ax.transAxes,
)

# useful horizonal positions
end_of_inept = LEFT_PAD + 4 * NINETY_WIDTH + 2 * TAU_WIDTH
end_of_t1 = end_of_inept + 2 * HALF_T1_WIDTH + 3 * NINETY_WIDTH
acquisition = end_of_t1 + 3 * NINETY_WIDTH + 2 * TAU_WIDTH
add_pulse(ax, "top", LEFT_PAD)
add_pulse(ax, "top", LEFT_PAD + NINETY_WIDTH + TAU_WIDTH, "180")
add_pulse(ax, "bottom", LEFT_PAD + NINETY_WIDTH + TAU_WIDTH, "180")
add_pulse(ax, "top", LEFT_PAD + 3 * NINETY_WIDTH + 2 * TAU_WIDTH)
add_pulse(ax, "bottom", LEFT_PAD + 3 * NINETY_WIDTH + 2 * TAU_WIDTH)
add_pulse(ax, "top", end_of_inept + HALF_T1_WIDTH, "180")
add_pulse(ax, "top", end_of_inept + 2 * HALF_T1_WIDTH + 2 * NINETY_WIDTH)
add_pulse(ax, "bottom", end_of_inept + 2 * HALF_T1_WIDTH + 2 * NINETY_WIDTH)
add_pulse(ax, "top", end_of_t1 + TAU_WIDTH, "180")
add_pulse(ax, "bottom", end_of_t1 + TAU_WIDTH, "180")
add_pulse(ax, "top", end_of_t1 + 2 * TAU_WIDTH + 2 * NINETY_WIDTH)

ax.add_patch(
    Rectangle(
        (acquisition, BOTTOM_PAD), T2_WIDTH, 0.4 * CHANNEL_HEIGHT,
        transform=ax.transAxes, facecolor="#a0a0a0", edgecolor="none",
    )
)


dotted_line(ax, "bottom", LEFT_PAD + NINETY_WIDTH)
dotted_line(ax, "bottom", LEFT_PAD + 5 * NINETY_WIDTH + 2 * TAU_WIDTH + HALF_T1_WIDTH)
dotted_line(ax, "bottom", LEFT_PAD + 9 * NINETY_WIDTH + 4 * TAU_WIDTH + 2 * HALF_T1_WIDTH)

add_text(ax, LEFT_PAD + NINETY_WIDTH + (0.5 * TAU_WIDTH), BOTTOM_PAD + 0.5 * CHANNEL_HEIGHT, r"$\tau$")
add_text(ax, LEFT_PAD + 3 * NINETY_WIDTH + (1.5 * TAU_WIDTH), BOTTOM_PAD + 0.5 * CHANNEL_HEIGHT, r"$\tau$")
add_text(ax, LEFT_PAD + NINETY_WIDTH + (0.5 * TAU_WIDTH), BOTTOM_PAD + 1.5 * CHANNEL_HEIGHT + CHANNEL_GAP, r"$\tau$")
add_text(ax, LEFT_PAD + 3 * NINETY_WIDTH + (1.5 * TAU_WIDTH), BOTTOM_PAD + 1.5 * CHANNEL_HEIGHT + CHANNEL_GAP, r"$\tau$")
add_text(ax, end_of_inept + 0.5 * HALF_T1_WIDTH +  0.5 * NINETY_WIDTH, BOTTOM_PAD + 0.5 * CHANNEL_HEIGHT, r"$\frac{t_1}{2}$")
add_text(ax, end_of_inept + 1.5 * HALF_T1_WIDTH + 1.5 * NINETY_WIDTH, BOTTOM_PAD + 0.5 * CHANNEL_HEIGHT, r"$\frac{t_1}{2}$")
add_text(ax, end_of_t1 + 0.5 * TAU_WIDTH, BOTTOM_PAD + 0.5 * CHANNEL_HEIGHT, r"$\tau$")
add_text(ax, end_of_t1 + 1.5 * TAU_WIDTH + 2 * NINETY_WIDTH, BOTTOM_PAD + 0.5 * CHANNEL_HEIGHT, r"$\tau$")
add_text(ax, end_of_t1 + 0.5 * TAU_WIDTH, BOTTOM_PAD + 1.5 * CHANNEL_HEIGHT + CHANNEL_GAP, r"$\tau$")
add_text(ax, end_of_t1 + 1.5 * TAU_WIDTH + 2 * NINETY_WIDTH, BOTTOM_PAD + 1.5 * CHANNEL_HEIGHT + CHANNEL_GAP, r"$\tau$")
add_text(ax, acquisition + 0.5 * T2_WIDTH, BOTTOM_PAD + 0.2 * CHANNEL_HEIGHT, "decouple")
add_text(ax, end_of_inept - 0.5 * NINETY_WIDTH, BOTTOM_PAD + 2 * CHANNEL_HEIGHT + CHANNEL_GAP + 0.04, "$y$")
add_text(ax, end_of_t1 - 0.5 * NINETY_WIDTH, BOTTOM_PAD + CHANNEL_HEIGHT + 0.04, r"$x,y$")

tp = np.linspace(acquisition, acquisition + T2_WIDTH, 256)
fid = (BOTTOM_PAD + CHANNEL_HEIGHT + CHANNEL_GAP) + (
    0.5 * CHANNEL_HEIGHT * np.cos(300 * (tp - acquisition)) *
    np.exp(np.linspace(0, -4, tp.size))
)
ax.plot(tp, fid, transform=ax.transAxes, color="k", lw=1.4)

fig.savefig("hsqc.pdf")
fig.savefig("hsqc.png")
