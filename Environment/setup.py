# setup.py
from setuptools import setup, find_packages

setup(
    name="SDSR-SU",
    version="0.1.0",
    description="Code for course: Machine Learning (SDSR)",
    author="Your Name",
    author_email="franko.hrzic@uniri.hr",
    packages=find_packages(),  
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "jupyterlab",
        "notebook",
        "tqdm",
        "ipywidgets",
        "torch",
        "torchvision",
        "torchinfo",
        "scikit-learn"
    ],
    entry_points={
        'console_scripts': [
        ],
    },

    python_requires=">=3.12",
)