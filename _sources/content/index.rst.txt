.. nmr_sims documentation master file, created by
   sphinx-quickstart on Tue Feb 15 14:04:30 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

nmr_sims: NMR Simulations
=========================

This is a python package to facilitate simulating liquid-state NMR experiments.
The capabilities are fairly limited:

* All simulations are performed using Hilbet space Cartesian basis sets
* Isotropic chemical shift and scalar coupling are the only interactions supported
* No relaxation model is provided

Perhaps more sophiostication will be added to a later version, but I cannot guarantee this.

This documentation is farily sparse at the moment. Hopefully you can get up-and-running
by looking at the :ref:`Setting Up a Simulation` page, and subsequently
the descriptions of the various modules in the :ref:`Reference`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   setup_simulation
   references/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
