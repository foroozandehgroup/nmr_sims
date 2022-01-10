# test.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 10 Jan 2022 17:53:23 GMT

import pytest
import numpy as np
from nmr_sims import CartesianBasis, Operator, SpinSystem


sq2 = np.sqrt(2)
sq3 = np.sqrt(3)
sq5 = np.sqrt(5)

E_HALF = Operator(0.5 * np.eye(2))
X_HALF = Operator(np.array([[0, 0.5], [0.5, 0]]))
Y_HALF = Operator(np.array([[0, -0.5j], [0.5j, 0]]))
Z_HALF = Operator(np.array([[0.5, 0], [0, -0.5]]))
MINUS_HALF = Operator(np.array([[0, 0], [1, 0]]))
PLUS_HALF = Operator(np.array([[0, 1], [0, 0]]))

X_ONE = (1 / sq2) * Operator(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
Y_ONE = (1 / sq2) * Operator(np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]))
Z_ONE = Operator(np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]))

X_THREE_HALF = Operator(0.5 * np.array(
    [[0, sq3, 0, 0], [sq3, 0, 2, 0], [0, 2, 0, sq3], [0, 0, sq3, 0]]
))
Y_THREE_HALF = Operator(-0.5j * np.array(
    [[0, sq3, 0, 0], [-sq3, 0, 2, 0], [0, -2, 0, sq3], [0, 0, -sq3, 0]]
))
Z_THREE_HALF = Operator(0.5 * np.array(
    [[3, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -3]]
))
X_FIVE_HALF = Operator(0.5 * np.array(
    [
        [0, sq5, 0, 0, 0, 0],
        [sq5, 0, 2 * sq2, 0, 0, 0],
        [0, 2 * sq2, 0, 3, 0, 0],
        [0, 0, 3, 0, 2 * sq2, 0],
        [0, 0, 0, 2 * sq2, 0, sq5],
        [0, 0, 0, 0, sq5, 0],
    ]
))

X1Y2 = Operator(np.array(
    [
        [0, 0, 0, -0.5j],
        [0, 0, 0.5j, 0],
        [0, -0.5j, 0, 0],
        [0.5j, 0, 0, 0],
    ]
))
Z1E2 = Operator(np.array(
    [
        [0.5, 0, 0, 0],
        [0, 0.5, 0, 0],
        [0, 0, -0.5, 0],
        [0, 0, 0, -0.5],
    ]
))
E1Z2 = Operator(np.array(
    [
        [0.5, 0, 0, 0],
        [0, -0.5, 0, 0],
        [0, 0, 0.5, 0],
        [0, 0, 0, -0.5],
    ]
))


class TestOperator:
    def test_init(self):
        op = Operator(np.array([[0, 0.5], [0.5, 0]]))
        assert isinstance(op, Operator)
        assert op.matrix.dtype == "complex"

    def test_neg(self):
        assert (
            X_HALF == -Operator(np.array([[0, -0.5], [-0.5, 0]]))
        )

    def test_add(self):
        assert X_HALF + E_HALF == Operator(0.5 * np.ones((2, 2)))

        fails = (X_HALF.matrix, 0.5)
        for fail in fails:
            with pytest.raises(TypeError):
                X_HALF + fail

    def test_sub(self):
        assert X_HALF - E_HALF == Operator(np.array([[-0.5, 0.5], [0.5, -0.5]]))

        fails = (E_HALF.matrix, 0.5)
        for fail in fails:
            with pytest.raises(TypeError):
                X_HALF - fail

    def test_mul_rmul(self):
        test_values = (2, 2., 2j)
        test_results = (
            Operator(np.array([[0, 1], [1, 0]])),
            Operator(np.array([[0, 1], [1, 0]])),
            Operator(np.array([[0, 1j], [1j, 0]])),
        )

        for value, result in zip(test_values, test_results):
            mul_res = X_HALF * value
            assert isinstance(mul_res, Operator)
            assert mul_res == result
            rmul_res = value * X_HALF
            assert mul_res == rmul_res

        # (Operator * Operator) and (Operator * np.ndarray) not supported.
        # Don't think there is a case in MR when this would be needed?
        with pytest.raises(TypeError):
            X_HALF * X_HALF
        with pytest.raises(TypeError):
            X_HALF * X_HALF.matrix

    def test_matmul(self):
        assert X_HALF @ Y_HALF == 0.5j * Z_HALF
        assert Y_HALF @ X_HALF == -0.5j * Z_HALF

        # (Operator @ np.ndarray) not permitted.
        with pytest.raises(TypeError):
            X_HALF @ Y_HALF.matrix

    def test_power(self):
        assert X_HALF ** 2 == 0.5 * E_HALF

    def test_adjoint(self):
        assert Y_HALF.adjoint == Y_HALF
        assert (
            Operator(np.array([[1, 0.5j], [-1j, 1]])).adjoint ==
            Operator(np.array([[1, 1j], [-0.5j, 1]]))
        )

    def test_trace(self):
        assert X_HALF.trace == 0
        assert Y_HALF.trace == 0
        assert Z_HALF.trace == 0
        assert (
            Operator(np.array([[3, 2j, 1j], [0.5, 1 + 2j, 0], [1j, 2, 2j]])).trace ==
            4 + 4j
        )

    def test_commutator(self):
        rotate = lambda l, n: l[n:] + l[:n]
        for a, b, c in [rotate([X_HALF, Y_HALF, Z_HALF], i) for i in range(3)]:
            assert a.commutator(b) == 1j * c
            assert b.commutator(a) == -1j * c

        assert Z_HALF.commutator(PLUS_HALF) == PLUS_HALF
        assert Z_HALF.commutator(MINUS_HALF) == -MINUS_HALF

    def test_commutes_with(self):
        l_squared = X_HALF ** 2 + Y_HALF ** 2 + Z_HALF ** 2
        for i, op1 in enumerate((X_HALF, Y_HALF, Z_HALF)):
            assert op1.commutes_with(l_squared)
            for j, op2 in enumerate((X_HALF, Y_HALF, Z_HALF)):
                if i == j:
                    assert op1.commutes_with(op2)
                else:
                    assert not op1.commutes_with(op2)

    def test_kroenecker(self):
        x1y2 = X_HALF.kroenecker(Y_HALF)
        assert x1y2 == Operator(np.array(
            [
                [0, 0, 0, -0.25j],
                [0, 0, 0.25j, 0],
                [0, -0.25j, 0, 0],
                [0.25j, 0, 0, 0],
            ]
        ))

    def test_operator_generators(self):
        assert Operator.Ix(0.5) == X_HALF
        assert Operator.Iy(0.5) == Y_HALF
        assert Operator.Iz(0.5) == Z_HALF
        assert Operator.Iplus(0.5) == PLUS_HALF
        assert Operator.Iminus(0.5) == MINUS_HALF
        assert Operator.Ix(1) == X_ONE
        assert Operator.Iy(1) == Y_ONE
        assert Operator.Iz(1) == Z_ONE
        assert Operator.Ix(1.5) == X_THREE_HALF
        assert Operator.Iy(1.5) == Y_THREE_HALF
        assert Operator.Iz(1.5) == Z_THREE_HALF
        assert Operator.Ix(2.5) == X_FIVE_HALF


class TestCartesianBasis:
    def test_init(self):
        with pytest.raises(ValueError) as exc_info:
            CartesianBasis(I=0.2)
        assert "`I` should be a multiple of 1/2, but is 0.2." == str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            CartesianBasis(I=1.5, nspins=0)
        assert (
            "`nspins` should be an int greater than 0 but is 0." ==
            str(exc_info.value)
        )

        basis = CartesianBasis()
        assert basis.nspins == 1
        assert basis._Ix == X_HALF
        assert basis._Iy == Y_HALF
        assert basis._Iz == Z_HALF
        assert basis._E == E_HALF

        basis = CartesianBasis(I=1.5, nspins=3)
        assert basis.nspins == 3
        assert basis._Ix == X_THREE_HALF
        assert basis._Iy == Y_THREE_HALF
        assert basis._Iz == Z_THREE_HALF

    def test_get(self):
        basis = CartesianBasis(nspins=2)
        with pytest.raises(ValueError) as exc_info:
            basis.get("1x2w")
        assert "Should satisfy the regex ^(\\d+(x|y|z))+$" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            basis.get("1x2y3z")
        assert "Spin 3 does not exist for basis of 2 spins." in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            basis.get("1x2y2z")
        assert "Spin 2 is repeated." in str(exc_info.value)

        assert basis.get("1x2y") == X1Y2
        assert basis.get("1z") == Z1E2
        assert basis.get("2z") == E1Z2


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
        assert "`spins` should solely have ints as keys." == str(exc_info.value)

    def test_nonconsecutive_int_keys(self):
        # spins dict has labels 2, 3. Not valid as no 1!
        spins = self.get_spins()
        spins[3] = spins[1]
        del spins[1]
        with pytest.raises(ValueError) as exc_info:
            SpinSystem(spins)
        assert (
            "The keys in `spins` should be consecutive ints, starting at 1." ==
            str(exc_info.value)
        )

    def test_invalid_spin_dict(self):
        # spin 1 does not have a shift defined
        spins = self.get_spins()
        del spins[1]["shift"]
        with pytest.raises(ValueError) as exc_info:
            SpinSystem(spins)
        assert (
            "\"shift\" and \"couplings\". This is not satisfied by spin 1." in
            str(exc_info.value)
        )

    def test_invalid_shift(self):
        # spin 2's shift is given by a string.
        spins = self.get_spins()
        spins[2]["shift"] = "40"
        with pytest.raises(TypeError) as exc_info:
            SpinSystem(spins)
        assert (
            "should be scalar values. This is not satisfied by spin 2." in
            str(exc_info.value)
        )

    def test_invalid_coupling_key(self):
        spins = self.get_spins()
        spins[1]["couplings"][1] = spins[1]["couplings"][2]
        del spins[1]["couplings"][2]
        with pytest.raises(ValueError) as exc_info:
            SpinSystem(spins)
        errmsg = str(exc_info.value)
        assert "spin 1: 1." in errmsg
        assert "no more than 2, and not 1." in errmsg

        spins = self.get_spins()
        spins[1]["couplings"][3] = spins[1]["couplings"][2]
        with pytest.raises(ValueError) as exc_info:
            SpinSystem(spins)
        errmsg = str(exc_info.value)
        assert "spin 1: 3." in errmsg
        assert "no more than 2, and not 1." in errmsg

    def test_contradictionary_couplings(self):
        spins = self.get_spins()
        spins[1]["couplings"][2] = 20
        with pytest.raises(ValueError) as exc_info:
            SpinSystem(spins)
        assert "between spins 1 and 2: 40.0 and 20.0." in str(exc_info.value)

    def test_init(self):
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

    def test_free_hamiltonian(self):
        spin_system = SpinSystem(self.get_spins())
        print(spin_system.free_hamiltonian)
