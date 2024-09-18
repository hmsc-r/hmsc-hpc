from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages
import shutil
import subprocess
import os

class CustomBuildExtCommand(build_ext):
    """Custom build command that runs Makefile to build the shared library."""

    def run(self):
        # Check if MAGMA_ROOT is set in the environment
        magma_root = os.environ.get('MAGMA_ROOT')
        if not magma_root:
            raise RuntimeError("MAGMA_ROOT environment variable is not set.")
        
        # Pass MAGMA_ROOT to the Makefile
        subprocess.check_call(['make', 'all', f'MAGMA_ROOT={magma_root}'])
        super().run()


class CustomInstallCommand(install):
    def run(self):
        self.run_command('build_ext')
        install.run(self)
        # Copy the .so file to the desired location
        source = os.path.join(os.path.dirname(__file__), 'hmsc', 'magma_cholesky', 'magma_cholesky.so')
        destination = os.path.join(self.install_lib, 'hmsc', 'magma_cholesky', 'magma_cholesky.so')
        shutil.copyfile(source, destination)

        super().run()

setup(
    name="hmsc",
    version="0.1.1",
    author="[removed for review]",
    license='GPLv3+',
    packages=find_packages(include=['hmsc', 'hmsc.*']),
    package_data={
        'hmsc.magma_cholesky': ['magma_cholesky.so'],
    },
    cmdclass={
        'build_ext': CustomBuildExtCommand,
        'install': CustomInstallCommand,
    },
    install_requires=[
        'pyreadr',
        'tensorflow-rocm==2.14.0.600',
        'tensorflow-probability==0.22.0',
        'numpy',
        'pandas',
        'ujson',
    ]
)
