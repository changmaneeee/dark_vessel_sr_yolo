"""Setup script for AIS-SAT-PIPELINE package"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "AIS-SAT-PIPELINE: SR-Detection Feature Fusion for VLEO Satellite Ship Detection"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="ais-sat-pipeline",
    version="0.1.0",
    author="AIS-SAT Team",
    author_email="your.email@example.com",
    description="SR-Detection Feature Fusion Pipeline for VLEO Satellite Ship Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ais-sat-pipeline",
    packages=find_packages(exclude=["tests", "scripts", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ais-sat-train=scripts.train:main",
            "ais-sat-eval=scripts.evaluate:main",
            "ais-sat-infer=scripts.inference:main",
        ],
    },
)
