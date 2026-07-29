#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="steakhouse_ai",
    version="0.0.1",
    description="Cooperative multi-agent environment based on Steakhouse",
    author="Ya-Chuan (Sophie) Hsu",
    author_email="yachuanh@usc.edu",
    packages=find_packages("."),
    keywords=["Steakhouse", "AI", "Reinforcement Learning"],
    package_data={
        "overcooked_ai_py": [
            "data/**/*",
            "configs/**/*",
            "assets/**/*",
            "images/**/*",
        ]
    },
    install_requires=[
        "hydra-core",
        "omegaconf",
        "gymnasium",
        "torch",
        "numpy",
        "wandb",
        "pygame",
        "opencv-python",
    ],
)
