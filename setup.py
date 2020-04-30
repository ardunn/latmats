#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="latmats",
    version="0.1",
    description="General Latent Representations of Materials From Text Mining (LatMats)",
    long_description=readme,
    author="Materials Scholar Development Team",
    author_email="amalietrewartha@lbl.gov, ardunn@lbl.gov",
    url="https://github.com/ardunn/latmats",
    license=license,
    packages=find_packages(),
    # test_suite="matscholar_web",
    # tests_require="tests",
)