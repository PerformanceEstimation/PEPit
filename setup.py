import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PEPit",
    version="0.0.1",
    author="TBC",
    author_email="TBC",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bgoujaud/PEPit",
    project_urls={
        "Documentation": "https://github.com/bgoujaud/PEPit/documentation",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "PEPit"},
    packages=setuptools.find_packages(where="PEPit"),
    python_requires=">=3.6",
)
