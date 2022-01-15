# test.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 12 Jan 2022 15:16:06 GMT

import pytest
import numpy as np
from nmr_sims import CartesianBasis, Operator, SpinSystem, nuclei



class TestSpinSystem:

    @staticmethod
    def get_spins():
        return {
            1: {
                "shift": 1000,
                "couplings": {
                    2: 40,
                },
            },
            2: {
                "shift": 500,
                "couplings": {
                    1: 40,
                },
            },
        }

    def test_keys_not_ints(self):
        # Key is a str, when all should be ints
        spins = self.get_spins()
        spins["1"] = spins[1]
        del spins[1]
        with pytest.raises(ValueError) as exc_info:
            SpinSystem(spins)
        assert str(exc_info.value) == (
            "`spins` should be a dict, and it's keys should be consecutive ints, "
            "starting at 1."
        )

    def test_nonconsecutive_int_keys(self):
        # spins dict has labels 2, 3. Not valid as no 1!
        spins = self.get_spins()
        spins[3] = spins[1]
        del spins[1]
        with pytest.raises(ValueError) as exc_info:
            SpinSystem(spins)
        assert str(exc_info.value) == (
            "`spins` should be a dict, and it's keys should be consecutive ints, "
            "starting at 1."
        )

    def test_invalid_spin_dict(self):
        # spin 1 does not have a shift defined
        spins = self.get_spins()
        del spins[1]["shift"]
        with pytest.raises(ValueError) as exc_info:
            SpinSystem(spins)
        assert str(exc_info.value) == (
            "Each value in `spins` should be a dict with the keys \"shift\" and "
            "(optional) \"couplings\". This is not satisfied by spin 1."
        )

    def test_invalid_shift(self):
        # spin 2's shift is given by a string.
        spins = self.get_spins()
        spins[2]["shift"] = "40"
        with pytest.raises(TypeError) as exc_info:
            SpinSystem(spins)
        assert str(exc_info.value) == (
            "\"shift\" entries should be scalar values. This is not satisfied by "
            "spin 2."
        )

    def test_invalid_coupling_key(self):
        spins = self.get_spins()
        spins[1]["couplings"][1] = spins[1]["couplings"][2]
        del spins[1]["couplings"][2]

        expected_errmsg = (
            "`spins[1][\"couplings\"]` should be a dict, and it's keys should "
            "be positive ints, that are no greater than 2. 1 is not permitted."
        )

        with pytest.raises(ValueError) as exc_info:
            SpinSystem(spins)
        assert str(exc_info.value) == expected_errmsg

        spins = self.get_spins()
        spins[1]["couplings"][3] = spins[1]["couplings"][2]
        with pytest.raises(ValueError) as exc_info:
            SpinSystem(spins)
        assert str(exc_info.value) == expected_errmsg

    def test_contradictionary_couplings(self):
        spins = self.get_spins()
        spins[1]["couplings"][2] = 20
        with pytest.raises(ValueError) as exc_info:
            SpinSystem(spins)
        assert "between spins 1 and 2: 40.0 and 20.0." in str(exc_info.value)

    def test_init(self):
        # Case without coupling specified
        spins = {1: {"shift": 1000}}
        spin_system = SpinSystem(spins)
        assert np.array_equal(spin_system.shifts, np.array([1000]))
        assert np.array_equal(spin_system.couplings, np.array([[0]]))
        assert spin_system.get("1x") == X_HALF

        spins = self.get_spins()
        spin_system = SpinSystem(spins)
        assert spin_system.nspins == 2
        assert np.array_equal(spin_system.shifts, np.array([1000, 500]))
        assert np.array_equal(spin_system.couplings, np.array([[0, 40], [40, 0]]))
        assert spin_system.get("1x2y") == X1Y2

    def test_edit_shift(self):
        spin_system = SpinSystem(self.get_spins())
        spin_system.edit_shift(1, -500)
        spin_system.edit_shift(2, 600)
        assert np.array_equal(spin_system.shifts, np.array([-500, 600]))

        with pytest.raises(ValueError) as exc_info:
            spin_system.edit_shift(3, 1000)
        assert "greater than 0 and no more than 2" in str(exc_info.value)

        with pytest.raises(TypeError) as exc_info:
            spin_system.edit_shift(1, "-500")
        assert str(exc_info.value) == "`value` should be a scalar value."

    def test_edit_copuling(self):
        spin_system = SpinSystem(self.get_spins())
        spin_system.edit_coupling(1, 2, 60)
        assert np.array_equal(spin_system.couplings, np.array([[0, 60], [60, 0]]))

        with pytest.raises(ValueError) as exc_info:
            spin_system.edit_coupling(1, 1, 60)
        assert str(exc_info.value) == "`spin1` and `spin2` cannot match."
