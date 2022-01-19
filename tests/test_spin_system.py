# test_spin_system.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 17 Jan 2022 18:14:56 GMT

from nmr_sims import spin_system, experimental


def test():
    spins = {
        1: {
            "shift": 4,
            "couplings": {
                2: 32,
            },
        },
        2: {
            "shift": 7,
            # "nucleus": "2H",
        }
    }

    exp = experimental.Experimental(
        channels=["1H"],
        sweep_widths=[10000],
        offsets=[5000],
    )
    ss = spin_system.SpinSystem(spins)
    ss.set_conditions(exp)
    print(ss.pulse("1H"))
