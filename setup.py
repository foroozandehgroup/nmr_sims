from setuptools import setup, find_packages

setup(
    name='nmr_sims',
    version='0.0.1',
    packages=find_packages(include=['nmr_sims']),
    install_requires=[
        'numpy',
        'scipy',
    ],
    test_require=['pytest'],
    setup_requires=['flake8'],
)
