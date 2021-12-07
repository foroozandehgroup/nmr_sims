import numpy as np
from nmr_sims import Operator


class Test:
    @staticmethod
    def _check_close(a, b):
        return np.allclose(a, b, rtol=0.0, atol=1E-10)

    def test_operator(self):
        sqrt2 = np.sqrt(2)
        sqrt3 = np.sqrt(3)
        sqrt5 = np.sqrt(5)
        Ix_half = Operator.Ix(0.5)
        assert self._check_close(
            Ix_half.matrix, np.array([[0.0, 0.5], [0.5, 0.0]], dtype="complex")
        )
        Iy_half = Operator.Iy(0.5)
        assert self._check_close(
            Iy_half.matrix, np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype="complex")
        )
        Iz_half = Operator.Iz(0.5)
        assert self._check_close(
            Iz_half.matrix, np.array([[0.5, 0.0], [0.0, -0.5]], dtype="complex")
        )
        Iplus_half = Operator.Iplus(0.5)
        assert self._check_close(
            Iplus_half.matrix, np.array([[0.0, 1.0], [0.0, 0.0]], dtype="complex")
        )
        Iminus_half = Operator.Iminus(0.5)
        assert self._check_close(
            Iminus_half.matrix, np.array([[0.0, 0.0], [1.0, 0.0]], dtype="complex")
        )

        Ix_one = Operator.Ix(1)
        assert self._check_close(
            Ix_one.matrix,
            (1 / sqrt2) * np.array([
                [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]
            ], dtype="complex"),
        )
        Iy_one = Operator.Iy(1)
        assert self._check_close(
            Iy_one.matrix,
            (1j / sqrt2) * np.array([
                [0.0, -1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, 0.0]
            ], dtype="complex"),
        )
        Iz_one = Operator.Iz(1)
        assert self._check_close(
            Iz_one.matrix,
            np.array([
                [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]
            ], dtype="complex"),
        )

        Ix_three_halves = Operator.Ix(1.5)
        assert self._check_close(
            Ix_three_halves.matrix,
            0.5 * np.array([
                [0.0, sqrt3, 0.0, 0.0],
                [sqrt3, 0.0, 2.0, 0.0],
                [0.0, 2.0, 0.0, sqrt3],
                [0.0, 0.0, sqrt3, 0.0],
            ], dtype="complex"),
        )
        Iy_three_halves = Operator.Iy(1.5)
        assert self._check_close(
            Iy_three_halves.matrix,
            -0.5j * np.array([
                [0.0, sqrt3, 0.0, 0.0],
                [-sqrt3, 0.0, 2.0, 0.0],
                [0.0, -2.0, 0.0, sqrt3],
                [0.0, 0.0, -sqrt3, 0.0],
            ], dtype="complex"),
        )
        Iz_three_halves = Operator.Iz(1.5)
        assert self._check_close(
            Iz_three_halves.matrix,
            0.5 * np.array([
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, -3.0],
            ], dtype="complex"),
        )

        Ix_five_halves = Operator.Ix(2.5)
        assert self._check_close(
            Ix_five_halves.matrix,
            0.5 * np.array([
                [0.0, sqrt5, 0.0, 0.0, 0.0, 0.0],
                [sqrt5, 0.0, 2 * sqrt2, 0.0, 0.0, 0.0],
                [0.0, 2 * sqrt2, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 2 * sqrt2, 0.0],
                [0.0, 0.0, 0.0, 2 * sqrt2, 0.0, sqrt5],
                [0.0, 0.0, 0.0, 0.0, sqrt5, 0.0],
            ], dtype="complex"),
        )

    def test_commutator(self):
        Ix = Operator.Ix(0.5)
        assert np.array_equal(
            Ix.matrix, np.array([[0.0, 0.5], [0.5, 0.0]], dtype="complex")
        )
        Iy = Operator.Iy(0.5)
        Iz = Operator.Iz(0.5)
        Iplus = Operator.Iplus(0.5)
        # [Iz, I+] = I+
        assert Iz.commutator(Iplus) == Iplus
        # [Ix, Iy] = iIz
        rotate = lambda l, n : l[n:] + l[:n]
        for a, b, c in [rotate([Ix, Iy, Iz], i) for i in range(3)]:
            assert a.commutator(b) == 1j * c
            assert b.commutator(a) == -1j * c

    def test_matmul(self):
        Iz = Operator.Iz(0.5)
        Iz_squared = Operator(0.5, matrix=np.array([[0.25, 0.0], [0.0, 0.25]]))
        Iz_cubed = Operator(0.5, matrix=np.array([[0.125, 0.0], [0.0, -0.125]]))
        assert Iz @ Iz == Iz_squared
        assert Iz ** 2 == Iz_squared
        assert Iz @ Iz @ Iz == Iz_cubed
        assert Iz ** 3 == Iz_cubed

    def test_kroenecker(self):
        z = Operator.Iz(0.5)
        zz = 0.25 * np.diag(np.array([1, -1, -1, 1], dtype="complex"))
        assert z.kroenecker(z) == Operator(0.5, nspins=2, matrix=zz)
