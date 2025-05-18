# setup.py
#
# Setup Instructions for MML Library
# From MML Library by Nathmath

from setuptools import setup, find_packages

setup(
    name="MML",
    version="0.0.4.0",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "torch", "pandas", "matplotlib",
                      "lzma", "gzip", "pickle", "uuid", "copy"],
    author="DOF Studio",
    author_email="dof.hbx@gmail.com",
    description="My Machine Learning (MML) Library. A hybrid backend (numpy or torch) machine learning and deep learning framework coding from scratch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dof-studio/MML/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License Version 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
