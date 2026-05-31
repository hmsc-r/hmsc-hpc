from setuptools import setup, find_packages
setup(
  name='hmsc',
  version='0.2.0',
  author='Gleb Tikhonov, Anis Ur Rahman, Tuomas Rossi, Jari Oksanen, Otso Ovaskainen',
  author_email='gleb.tikhonov@helsinki.fi, otso.ovaskainen@jyu.fi',
  maintainer='Gleb Tikhonov',
  maintainer_email='gleb.tikhonov@helsinki.fi',
  url='https://github.com/hmsc-r/hmsc-hpc',
  license='GPLv3+',
  packages=find_packages(include=['hmsc', 'hmsc.*']),
  install_requires=[
    'numpy>=2.0.0',
    'pandas>=2.1.0',
    'rdata>=1.0.0',
    'scipy>=1.11.0',
    'tensorflow[and-cuda]>=2.16.0',
    'tensorflow-probability[tf]>=0.24.0',
  ]
)
