from setuptools import setup, find_packages

setup(
    name="hmsc",
    version="0.1.9.dev1+io",
    author="[removed for review]",
    license="GPLv3+",
    packages=find_packages(include=["hmsc", "hmsc.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "rdata>=1.0.0",
        "scipy",
        "tensorflow",
        "tensorflow-probability[tf]",
    ],
)
