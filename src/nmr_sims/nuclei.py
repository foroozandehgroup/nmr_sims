# nuclei.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 16 Feb 2022 10:59:23 GMT

r"""The module enables specification of nuclei for use in generating spin systems
(see :py:mod:`nmr_sims.spin_system`). A general :py:class:`Nucleus` object allows
any nucleus (even imaginary ones) to be created. As well as this, there are numerous
pre-defined nuclei provided by :py:data:`supported_nuclei`.
"""


class Nucleus:
    """An object representing the key properties of a nucleus for NMR simulation."""

    def __init__(self, name: str, gamma: float, multiplicity: int) -> None:
        r"""Create a ``Nucelus`` instance.

        Parameters
        ----------

        name
            The name of the nucleus. This can be anything, but by convention it is
            probably a good idea to use a ``str`` of the form ``"<mass><element>"``,
            such as ``"1H"``, ``"13C"``, etc.

        gamma
            The gyromagnetic ratio of the nucleus in units of :math:`\mathrm{rad}
            \mathrm{s}^{-1}\mathrm{T}^{-1}` (i.e. in accordance with the values in
            the first column in the table found at
            `this link
            <https://en.wikipedia.org/wiki/Gyromagnetic_ratio#For_a_nucleus>`_
            multiplied by :math:`10^6`).

        multiplicity
            The number of eigenstates produced as a result of the Zeeman effect. This
            is equivalent to :math:`2I + 1`, where :math:`I` in the nuclear spin quantum
            number.
        """
        self.__dict__.update(locals())

    def __str__(self):
        return self.name

    @property
    def spin(self) -> float:
        """Return the spin quantum number of the nucleus.

        This is simply :math:`(M - 1) / 2`, where :math:`M` is the multiplicity.
        """
        return round((self.multiplicity - 1) / 2, 1)


# All from NMR Enc. 1996
#:
supported_nuclei = {
    "1H": Nucleus("1H", 2.6752218744e8, 2),
    "2H": Nucleus("2H", 4.10662791e7, 3),
    "13C": Nucleus("13C", 6.728284e7, 2),
    "15N": Nucleus("15N", -2.71261804e7, 2),
    "19F": Nucleus("19F", 2.518148e8, 2),
    "31P": Nucleus("31P", 1.08394e7, 2),
}
