# test_operators.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 13 Jan 2022 17:28:59 GMT

import pytest
import numpy as np
from nmr_sims._operators import Operator, CartesianBasis

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


