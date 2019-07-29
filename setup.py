import setuptools

setuptools.setup(
    name="SpaCell",
    version="0.0.1",
    author="Xiao Tan, Andrew Su, Quan Nguyen",
    author_email="xiao.tan@uq.edu.au, a.su@uq.net.au, quan.nguyen@imb.uq.edu.au",
    description="SpaCell Package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BiomedicalMachineLearning/Spacell.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
