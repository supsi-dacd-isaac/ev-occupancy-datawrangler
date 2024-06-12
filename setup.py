import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evoccupancydatawrangler",
    version="0.1",
    author="lorenz",
    author_email="lorenz@lorenz",
    description="Query and process data from the Swiss EV occupancy database",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/supsi-dacd-isaac/ev-occupancy-datawrangler",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNUv3",
        "Operating System :: OS Independent",
    ],
    install_requires=["absl-py >= 2.1.0",
                      "influxdb >= 5.3.2",
                      "matplotlib >= 3.9.0",
                      "numpy >= 1.26.4",
                      "pandas >= 2.2.2",
                      "tqdm >= 4.66.4",
                      "urllib3 >= 2.2.1",
                      ],
    python_requires='>=3.10',
)