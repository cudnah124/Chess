from setuptools import setup, find_packages

setup(
    name="chess_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "python-chess>=1.999",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
)
