from setuptools import setup

setup(
    name="orbital_dynamics",
    version="0.1",
    description="a collection of functions that helps calculate Kepler orbits around various planets in our Solar System",
    author="Boaz",
    author_email="boaz@lulav.space",
    packages=["kepler_dymnamics"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)