# nmr\_sims

NMR simulations in Python by Simon.

Some rather simple simulations of basic NMR pulse sequences.

Everything uses the Zeeman basis in Hilbert space, so nothing too fancy is possible.

Currently have pulse-acquire and 2DJ code available.

If interested in checking it out, take a look at (and run)
`nmr_sims/experiments/pa.py` and `nmr_sims/experiments/jres.py` for a couple of
examples.

To install: Clone this repo, activate a venv, run `pip install -e .` when inside
the repo's root directory. You might need to manually install numpy, scipy
and matplotlib to the venv too, I haven't checked whether this is done automatically.
