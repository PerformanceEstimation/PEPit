import os
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = "{{VERSION_PLACEHOLDER}}"

# Fallback for direct Git installation (where 'sed' replacement is not performed)
if version.startswith("{{"):
    # Default to 0.0.1.dev0 to ensure 'pip install' doesn't crash
    version = os.environ.get("PEPIT_VERSION", "0.0.1.dev0")

setuptools.setup(
    name="PEPit",
    version=version,
    author="Baptiste Goujaud, Céline Moucer, Julien Hendrickx, Francois Glineur, Adrien Taylor and Aymeric Dieuleveut",
    author_email="baptiste.goujaud@gmail.com",
    description="PEPit is a package that allows users "
                "to pep their optimization algorithms as easily as they implement them",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["cvxpy>=1.1.17", "pandas>=1.0.0", "sympy>=1.10"],
    url="https://github.com/PerformanceEstimation/PEPit",
    project_urls={
        "Documentation": "https://pepit.readthedocs.io/en/{}/".format(version),
    },
    download_url="https://github.com/PerformanceEstimation/PEPit/archive/refs/tags/{}.tar.gz".format(version),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=[pkg for pkg in setuptools.find_packages() if pkg == "PEPit" or pkg.startswith("PEPit.")],
    python_requires=">=3.9",
)
