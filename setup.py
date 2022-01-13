import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PEPit",
    version="0.0.2",
    author="Baptiste Goujaud, Céline Moucer, Julien Hendrickx, Francois Glineur, Adrien Taylor and Aymeric Dieuleveut",
    author_email="baptiste.goujaud@gmail.com",
    description="PEPit is a package that allows users "
                "to pep their optimization algorithms as easily as they implement them",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["cvxpy>=1.1.17"],
    url="https://github.com/bgoujaud/PEPit",
    project_urls={
        "Documentation": "https://github.com/bgoujaud/PEPit/docs",
    },
    download_url='https://github.com/bgoujaud/PEPit/archive/refs/tags/0.0.1.tar.gz',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=[element for element in setuptools.find_packages() if element[:5] == 'PEPit'],
    python_requires=">=3.6",
)
