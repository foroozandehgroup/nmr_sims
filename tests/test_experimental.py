# test_experimental.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 13 Jan 2022 17:31:26 GMT

from nmr_sims. experimental import Experimental


def test_experimental():
    experimental = Experimental(
        channels={
            1: {
                "nucleus": "1H",
                "sweep_width": "10ppm",
                "offset": "4ppm",
                "points": 2048,
            },
            2: {
                "nucleus": "13C",
                "sweep_width": "250ppm",
                "offset": "120ppm",
                "points": 128
            },
        },
        temperature="25C",
        field="800MHz",
    )
