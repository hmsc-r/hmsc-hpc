from setuptools import setup, find_packages


setup(
  name='hmsc',
  version='0.3.3',
  author='Gleb Tikhonov, Anis Ur Rahman, Tuomas Rossi, Jari Oksanen, Otso Ovaskainen',
  author_email='gleb.tikhonov@helsinki.fi, otso.ovaskainen@jyu.fi',
  maintainer='Gleb Tikhonov',
  maintainer_email='gleb.tikhonov@helsinki.fi',
  url='https://github.com/hmsc-r/hmsc-hpc',
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
