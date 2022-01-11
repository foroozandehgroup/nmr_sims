# nuclei.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 11 Jan 2022 17:36:23 GMT

from dataclasses import dataclass


@dataclass
class Nucleus:
    name: str
    gamma: float
    multiplicity: int

    def __str__(self):
        return self.name


# All from NMR Enc. 1996
supported_nuclei = {
    "1H": Nucleus("1H", 2.6752218744e8, 2),
    "2H": Nucleus("2H", 4.10662791e7, 3),
    "13C": Nucleus("13C", 6.728284e7, 2),
    "15N": Nucleus("15N", -2.71261804e7, 2),
    "19F": Nucleus("19F", 2.518148e8, 2),
    "31P": Nucleus("31P", 1.08394e7, 2),
}
