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
        'tensorflow[and-cuda]',
        'tensorflow-probability[tf]',
        'ujson',
    ]
)
