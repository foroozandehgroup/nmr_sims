# test_experimental.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 13 Jan 2022 17:31:26 GMT

from nmr_sims. experimental import Experimental


def test_experimental():
    experimental = Experimental(
        channels=["1H", "13C"],
        sweep_widths=[10000, 100000],
        field="800MHz",
        temperature="25C",
    )
