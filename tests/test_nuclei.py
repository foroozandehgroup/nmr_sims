# test_nuclei.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 12 Jan 2022 15:15:22 GMT

from nmr_sims import nuclei


def test_nuclei():
    novel_nucleus = nuclei.Nucleus("195Pt", 5.8385e7, 2)
    assert novel_nucleus.name == str(novel_nucleus) == "195Pt"
    assert novel_nucleus.gamma == 5.8385e7
    assert novel_nucleus.multiplicity == 2
    assert novel_nucleus.spin == 0.5
    assert nuclei.supported_nuclei["13C"] == nuclei.Nucleus("13C", 6.728284e7, 2)
