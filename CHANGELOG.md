# Changelog

## 0.0.1

**N.B. Version 0.0.1 is only compatible with Python3.10 and above. However, any Python3.7 installation or above will be able to
install it. You will get errors if you try using nmr\_sims-0.0.1 with any Python version other than 3.10. You should use a newer
version of nmr\_sims, which are actually compatible with 3.7 and above.**

First version.

- All simulations are performed in Hilbert space using the Zeeman (Cartesian) basis.
- Only isotropic interactions (isotropic chemical shift and scalar couplings) are supported.

This version provides support for:

- Generation of Cartesian basis spin operators for arbitrarily sized spin systems of any spin.
- Generation of Spin systems based on chemical shift and scalar couplings.
- Specification of basic experimental parameters (Temperature, field strength, sweep width, offset, number of points).
- Simulation of pulse-acquire and J-resolved spectroscopy experiments out-of-the-box.

## 0.0.2

`nmr_sims` is now functional on Python versions 3.7 and above. Other than that, its the same as `0.0.1`.

## 0.0.3

HQSC Experiments can now be simulated.
Some corrections to spin Hamiltonian definitions.

