# test_sanity.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 13 Jan 2022 17:30:30 GMT

import pytest

from nmr_sims import _sanity, nuclei


class TestCheckers:
    proton = nuclei.supported_nuclei["1H"]
    def test_process_temperature(self):
        assert _sanity.process_temperature(300) == 300.
        assert _sanity.process_temperature("300K") == 300.
        assert _sanity.process_temperature("300.5K") == 300.5
        assert _sanity.process_temperature(".5K") == 0.5
        assert _sanity.process_temperature("25C") == 298.15
        assert _sanity.process_temperature(".5C") == 273.65

    def test_process_field(self):
        assert _sanity.process_field(11.74) == 11.74
        assert _sanity.process_field("11.74T") == 11.74
        assert round(_sanity.process_field("500MHz"), 2) == 11.74
        assert round(_sanity.process_field("500.MHz"), 2) == 11.74
        assert round(_sanity.process_field("500.0MHz"), 2) == 11.74

    def test_process_sweep_width(self):
        assert _sanity.process_sweep_width(5000, self.proton, 11.743) == 5000.
        assert _sanity.process_sweep_width("5000Hz", self.proton, 11.743) == 5000.
        assert round(_sanity.process_sweep_width("10ppm", self.proton, 11.743)) == 5000.

    def test_process_offset(self):
        assert _sanity.process_offset(2000, self.proton, 11.743) == 2000.
        assert _sanity.process_offset("2000Hz", self.proton, 11.743) == 2000.
        assert round(_sanity.process_offset("4ppm", self.proton, 11.743)) == 2000.
        assert round(_sanity.process_offset("-4ppm", self.proton, 11.743)) == -2000.

    def test_process_channel(self):
        field = 11.743  # T
        channel = {
            "nucleus": "13C",
            "sweep_width": "10ppm",
            "offset": "-4ppm",
            "points": 2048
        }
        _sanity.process_channel(channel, field)
