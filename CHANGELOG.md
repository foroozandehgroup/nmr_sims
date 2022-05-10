# Changelog

## 0.0.1

*Feb 14, 2022*

**N.B. Version 0.0.1 is only compatible with Python3.10 and above. However, any
Python3.7 installation or above will be able to install it. You will get errors
if you try using nmr\_sims-0.0.1 with any Python version other than 3.10. You
should use a newer version of nmr\_sims, which are actually compatible with 3.7
and above.**

First version.

* All simulations are performed in Hilbert space using the Zeeman (Cartesian)
  basis.
* Only isotropic interactions (isotropic chemical shift and scalar couplings)
  are supported.

This version provides support for:

* Generation of Cartesian basis spin operators for arbitrarily sized spin
  systems of any spin.
* Generation of Spin systems based on chemical shift and scalar couplings.
* Specification of basic experimental parameters (Temperature, field strength,
  sweep width, offset, number of points).
* Simulation of pulse-acquire and J-resolved spectroscopy experiments
  out-of-the-box.

## 0.0.2

*Feb 14, 2022*

`nmr_sims` is now functional on Python versions 3.7 and above. Other than that,
its the same as `0.0.1`.

## 0.0.3

*Feb 28, 2022*

* HQSC Experiments can now be simulated.
* Some corrections to spin Hamiltonian definitions.

## 0.0.4

*Mar 23, 2022*

Fix of bug: Phase of FIDs from Pulse-acquire experiments was 180° off, and code
to derive the spectrum flipped the spectrum twice to give off the impression
that everything was fine! This has been corrected.

## 0.0.5

**RELEASE HAS BEEN YANKED: FORGOR TO INCLUDE** `networkx` **IN PACKAGE
REQUIREMENTS**

* **Due to the dependence of** `spins_system.SpinSystem.new_random` **on
  [networkx](https://networkx.org/documentation/stable/index.html), Python 3.8
  or higher is now required to use** ``nmr_sims``.
* Enable selection of the strength of exponential window applied to the FID.
  (`lb` argument for `fid` and `spectrum` methods).
* Random spin systems can now be generated using
  ``spins_system.SpinSystem.new_random``. Currently, only the generation of
  homonuclear ¹H systems is possible.

## 0.0.6

*May 10, 2022*

Error fix: Include `networkx` to list of package requirements.
