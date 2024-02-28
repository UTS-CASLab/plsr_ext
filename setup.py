from setuptools import setup, find_packages

setup(
    name="plsr_ext",
    version="0.0.1",
    author="Thanh Tung KHUAT",
    author_email="thanhtung09t2@gmail.com",
    description="A collection of extended versions of PLSR",
    packages=find_packages(),
    install_requires=[
        "numpy",
        # For machine learning components.
        "scikit-learn",
        "scipy",
    ],
)