import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = "0.3.0"

setuptools.setup(
    name="PEPit",
    version=version,
    author="Baptiste Goujaud, Céline Moucer, Julien Hendrickx, Francois Glineur, Adrien Taylor and Aymeric Dieuleveut",
    author_email="baptiste.goujaud@gmail.com",
    description="PEPit is a package that allows users "
                "to pep their optimization algorithms as easily as they implement them",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["cvxpy>=1.1.17"],
    url="https://github.com/PerformanceEstimation/PEPit",
    project_urls={
        "Documentation": "https://pepit.readthedocs.io/en/{}/".format(version),
    },
    download_url="https://github.com/PerformanceEstimation/PEPit/archive/refs/tags/{}.tar.gz".format(version),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=[element for element in setuptools.find_packages() if element[:5] == 'PEPit'],
    python_requires=">=3.7",
)
