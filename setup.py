from setuptools import setup, find_packages

setup(
    name="hmsc",
    version="0.1.9.dev2+io",
    author="[removed for review]",
    license="GPLv3+",
    packages=find_packages(include=["hmsc", "hmsc.*"]),
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.1.0",
        "rdata>=1.0.0",
        "scipy>=1.11.0",
        "tensorflow>=2.16.0",
        "tensorflow-probability[tf]>=0.24.0",
    ],
)
