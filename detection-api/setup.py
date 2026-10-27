from setuptools import find_packages, setup

setup(
    name="road-hazard",
    version="1.0.0",
    description="Offline road hazard detection CLI and API for Jetson Nano",
    packages=find_packages(),
    python_requires=">=3.6",
    entry_points={"console_scripts": ["road-hazard=road_hazard.cli:main"]},
)
