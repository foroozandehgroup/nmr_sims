# spin_system.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 25 Feb 2022 18:35:38 GMT

r"""This module provides the :py:class:`SpinSystem` class, allowing specification
of a specific spin system according to the nuclear idenitites, isotropic chemical
shifts, and J-couplings of each spin, along with the temperature and magnetic
field they are subjected to.

The ``SpinSystem`` class is used by all experiment simulation functions to derive
the equilibrium operator and Hamiltonians required.
"""

from typing import Dict, Iterable, Union
import numpy as np
from nmr_sims.nuclei import Nucleus
from nmr_sims._operators import CartesianBasis, Operator
from nmr_sims import _sanity


# Currently only applicable for homonuclear systems
class SpinSystem(CartesianBasis):
    """Object representing a particular spin system.

    This represents an ensemble of many identical spin systems subject to a
    specified temperature and magnetic field strength.
    """
    kB = 1.380649e-23
    hbar = 1.054571817e-34

    def __init__(
        self,
        spins: Dict[int, Dict],
        default_nucleus: Union[Nucleus, str] = "1H",
        field: Union[int, float, str] = "500MHz",
        temperature: Union[int, float, str] = "298K",
    ) -> None:
        r"""Create an instance ``SpinSystem``.

        Parameters
        ----------

        spins
            A dictionary with information on each spin that makes up the system.
            Below is an example of a valid format. This specifies a three proton
            (AMX) system, with the following parameters:

            .. math::

                \delta_1 = 5.4\ \mathrm{ppm}

                \delta_2 = 2.8\ \mathrm{ppm}

                \delta_3 = 0.5\ \mathrm{ppm}

                J_{12} = 4.6\ \mathrm{Hz}

                J_{13} = 7.4\ \mathrm{Hz}

                J_{23} = 2.7\ \mathrm{Hz}

            .. code:: python3

                spins = {
                    1: {
                        "shift": 5.4,
                        "couplings": {
                            2: 4.6,
                            3: 7.4,
                        },
                    },
                    2: {
                        "shift": 2.8,
                        "couplings": {
                            3: 2.7,
                        },
                    },
                    3: {
                        "shift": 0.5,
                    },
                }

            Note that there is no need to repeat coupling information. As the
            coupling between spins 1 and 2 has been specified within the entry
            for spin 1, there is no need to repeat this for spin 2. Similary,
            spin 3 has been given no explicit coupling information as this has
            already been provided in the specification of spins 1 and 2.

            Each entry in the ``spins`` dictionary should have an ``int`` as a key
            and a ``dict`` as its value. Within the each ``dict``, the following
            key-value pairs are permitted:

            * ``"shift"`` (necessary) - The isotropic chemical shift of the spin
              in ppm.
            * ``"couplings"`` (optional) - The scalar couplings between the spin
              and its coupling partners. This should be denoted by a ``dict`` with
              an ``int`` key specifying the coupling partner, and a ``float`` value,
              specifying the J-coupling in Hz.
            * ``"nucleus"`` (optional) - The identity of nucelus giving rise to
              the spin. If not specified, then ``default_nucleus`` will be used.
              Can be either an instance of :py:class:`nmr_sims.nuclei.Nucleus`
              of a string corresponding to one of the pre-defined nuclei.

        default_nucleus
            The nucleus identity of any spin in the ``spins`` dictionary that is not
            given an explicit ``"nucleus"`` key. By default, all unspecified nuclei
            will correspond to proton (¹H).

        field
            The magnetic field strength. The following inputs are valid:

            * A positive number (``int`` or ``float``). The field will be taken to be
              in Telsa.
            * A ``str`` satisfying the regex ``r"\d+T"``. Again, the field will be
              taken to be in Tesla
            * A ``str`` satisfying the regex ``r"\d+MHz"``. This corresponds to the
              field which induces the specified Larmor frequency for proton.

            By default, this is set as ``"500MHz"``, correponding to approximately
            11.74T.

        temperature
            The temperature. The following inputs are valid:

            * A positive number (``int`` or ``float``). The field will be taken to be
              in Kelvin.
            * A ``str`` satisfying the regex ``r"\d+K"``. Again, the field will be
              taken to be in Kelvin.
            * A ``str`` satisfying the regex ``r"-?\d+C"``. The temperature will be
              taken to be in Celcius.


        .. warning::

            **Current bug**

            Negative Celcius specifications (``"-\d+C"``) are not supported currently,
            and will raise an error if given. PLease manually convert to Kelvin.
        """
        self.temperature = _sanity.process_temperature(temperature)
        self.field = _sanity.process_field(field)
        self.spin_dict = _sanity.process_spins(spins, default_nucleus)
        super().__init__(
            [(spin.nucleus.multiplicity - 1) / 2 for spin in self.spin_dict.values()]
        )
        if not all([I == 0.5 for I in self.spins]):
            raise ValueError("Spin-1/2 nuclei are only supported currently!")

    @property
    def inverse_temperature(self) -> float:
        r"""Return the inverse temperature for the system.

        Given by :math:`\beta = \hbar / k_{\mathrm{B}} T`.
        """
        return self.hbar / (self.kB * self.temperature)

    @property
    def boltzmann_factor(self) -> Iterable[float]:
        r"""Return the Boltzmann factor for each spin in the spin system.

        This factor is given by :math:`2 \pi \beta \gamma B_0`. See
        :py:meth:`inverse_temperature` for the definition of :math:`\beta`.
        """
        return [
            2 * np.pi * self.inverse_temperature * spin.nucleus.gamma * self.field
            for spin in self.spin_dict.values()
        ]

    @property
    def basic_frequencies(self) -> np.ndarray:
        r"""Return the baisc (laboratory frame) frequencies of each spin.

        Given by :math:`-\gamma B_0 \left(1 + \delta \times 10^{-6}\right)`
        """
        return np.array([
            -spin.nucleus.gamma * self.field * (1 + (1e-6 * spin.shift))
            for spin in self.spin_dict.values()
        ])

    @property
    def rotframe_frequencies(self) -> np.ndarray:
        r"""Return the rotating frame frequencies for each spin.

        Given by :math:`-\gamma B_0 \left(\delta \times 10^{-6}\right)`
        """
        return np.array([
            -spin.nucleus.gamma * self.field * (1e-6 * spin.shift)
            for spin in self.spin_dict.values()
        ])

    @property
    def couplings(self) -> np.ndarray:
        couplings = np.zeros((self.nspins, self.nspins))
        for i, spin in self.spin_dict.items():
            for j, coupling in spin.couplings.items():
                if i < j:
                    couplings[i - 1, j - 1] = coupling

        return couplings + couplings.T

    def pulse(
        self, nucleus: str, phase: float = 0., angle: float = np.pi / 2
    ) -> Operator:
        """Return the operator for a pulse targeting a specific nucleus, with
        specified phase and flip angle.

        Parameters
        ----------

        nucleus
            The identity of the nucleus to be targeted. Should match the
            ``nucleus.name``.

        phase
            Desired phase of the pulse in radians.

        Angle
            Desired flip angle in radians.
        """
        operator = self.zero
        for i, spin in self.spin_dict.items():
            if spin.nucleus.name == nucleus:
                operator += (
                    np.cos(phase) * self.get(f"{i}x") +
                    np.sin(phase) * self.get(f"{i}y")
                )
        return operator.rotation_operator(angle)

    @property
    def equilibrium_operator(self) -> Operator:
        r"""Return the equilibrium operator of the spin system.

        Given by:

        .. math::

            \hat{\rho}_{\mathrm{eq}} =
            \frac{1}{N} \left(\hat{E} + 2 \pi \beta B_0 \sum_{i=1}^N
            \gamma_i \hat{I}_{iz}\right)

        Where :math:`N` is the number of spins, and :math:`\hat{E}` is the identity
        operator.
        """
        return (1 / self.nspins) * (
            self.identity + sum(
                [b * self.get(f"{i}z") for i, b in
                 enumerate(self.boltzmann_factor, start=1)],
                self.zero,
            )
        )

    def hamiltonian(self, offsets: Union[dict, None] = None, decouple=None) -> Operator:
        r"""Return the Hamiltonian for the spin system.

        Given by:

        .. math::

            \hat{H} = -B_0 \sum_{i=1}^N (\delta_i \times 10^{-6}) \gamma_i
            \hat{I}_{iz}
            + 2 \pi \sum_{i=1}^{N-1} \sum_{j=i+1}^N
            J_{ij} \left(
            \hat{I}_{ix} \hat{I}_{jx} +
            \hat{I}_{iy} \hat{I}_{jy} +
            \hat{I}_{iz} \hat{I}_{jz} \right)

        Parameters
        ----------

        offsets
            Specification of transmitter offsets for given nuclei, in units of Hz.
            As an example, the argument ``{"1H": 5000}`` would inorporate a
            transmtter offset of 5000Hz for proton.
        """
        H = self.zero
        frequencies = self.rotframe_frequencies
        couplings = self.couplings
        isotopes = [self.spin_dict[i].nucleus.name for i in range(1, self.nspins + 1)]
        for i, (isotope1, freq) in enumerate(zip(isotopes, frequencies), start=1):
            H += freq * self.get(f"{i}z")
            if i == self.nspins:
                break
            for j, (isotope2, coupling) in enumerate(zip(
                isotopes[i:],
                couplings[i:, i - 1]
            ), start=i + 1):
                if isotope1 == isotope2:
                    H += np.pi * coupling * (
                        self.get(f"{i}x{j}x") +
                        self.get(f"{i}y{j}y") +
                        self.get(f"{i}z{j}z")
                    )
                elif decouple not in (isotope1, isotope2):
                    H += np.pi * coupling * self.get(f"{i}z{j}z")
                # Decouple: don't include scalar coupling

        if offsets is not None:
            for nuc, off in offsets.items():
                H += 2 * np.pi * off * sum(
                    [self.get(f"{i}z")
                     for i, spin in self.spin_dict.items()
                     if spin.nucleus.name == nuc],
                    self.zero
                )

        return H

    def _get_sum(self, coord: str, nucleus: str) -> Operator:
        if nucleus is None:
            labels = list(range(1, self.nspins + 1))
        else:
            labels = [i for i, spin in self.spin_dict.items()
                      if spin.nucleus.name == nucleus]

        return sum([self.get(f"{i}{coord}") for i in labels], self.zero)

    def Ix(self, nucleus: Union[str, None] = None) -> Operator:
        r"""Return the :math:`\hat{I}_x` operator correpsonding to the system,
        with the option of specifying the nucelus to target.

        For a given nucleus :math:`n`, this is given by:

        .. math::

            \hat{I}_{x, n} = \sum_{i} \hat{I}_{ix},

        :math:`\forall i \in \{1, \cdots, N\}` satisfying the requirement that
        spin :math:`i` corresponds to nucelus :math:`n`.

        Parameters
        ----------

        nucleus
            The identity of the nucelus to target. If ``None``, no constraint will
            be put on the nucleus.
        """
        return self._get_sum("x", nucleus)

    def Iy(self, nucleus: Union[str, None] = None) -> Operator:
        r"""Return the :math:`\hat{I}_y` operator correpsonding to the system,
        with the option of specifying the nucelus to target.

        For a given nucleus :math:`n`, this is given by:

        .. math::

            \hat{I}_{y, n} = \sum_{i} \hat{I}_{iy},

        :math:`\forall i \in \{1, \cdots, N\}` satisfying the requirement that
        spin :math:`i` corresponds to nucelus :math:`n`.

        Parameters
        ----------

        nucleus
            The identity of the nucelus to target. If ``None``, no constraint will
            be put on the nucleus.
        """
        return self._get_sum("y", nucleus)

    def Iz(self, nucleus: Union[str, None] = None) -> Operator:
        r"""Return the :math:`\hat{I}_z` operator correpsonding to the system,
        with the option of specifying the nucelus to target.

        For a given nucleus :math:`n`, this is given by:

        .. math::

            \hat{I}_{z, n} = \sum_{i} \hat{I}_{iz},

        :math:`\forall i \in \{1, \cdots, N\}` satisfying the requirement that
        spin :math:`i` corresponds to nucelus :math:`n`.

        Parameters
        ----------

        nucleus
            The identity of the nucelus to target. If ``None``, no constraint will
            be put on the nucleus.
        """
        return self._get_sum("z", nucleus)
