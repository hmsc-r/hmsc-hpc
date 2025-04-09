from setuptools import setup, find_packages

setup(
    name="hmsc",
    version="0.1.4",
    author="[removed for review]",
    license='GPLv3+',
    packages=find_packages(include=['hmsc', 'hmsc.*']),
    install_requires=[
        'numpy',
        'pandas',
        'pyreadr',
        'scipy',
        'tensorflow',
        'tensorflow-probability[tf]',
        'ujson',
    ]
)
