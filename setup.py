from setuptools import setup

setup(
    name="hmsc",
    version="0.1.0",
    author="[removed for review]",
    license='GPLv3+',
    install_requires=[
        'numpy',
        'pandas',
        'pyreadr',
        'scipy',
        'tensorflow==2.15.0',
        'tensorflow-probability==0.23.0',
        'ujson',
    ]
)
