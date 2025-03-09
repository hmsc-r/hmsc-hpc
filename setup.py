from setuptools import setup, find_packages


setup(
    name="hmsc",
    version="0.3.0",
    author="[removed for review]",
    license='GPLv3+',
    packages=find_packages(include=['hmsc', 'hmsc.*']),
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
