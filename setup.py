from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.0.3b1'
DESCRIPTION = 'A torch-like package for building Predictive Coding Neural Networks.'
LONG_DESCRIPTION = 'Built with a torch-like API, this package allows for the creation of Predictive Coding Neural Networks.' \
                    'The package includes both fully-connected and convolutional layers, as well as different pre-defined models.' \
                    'The package also includes a set of tools for training and testing the models, with detailed logging.'

# Setting up
setup(
    name="pclib",
    version=VERSION,
    author="Joe Griffith",
    author_email="joeagriffith@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['torch', 'torchvision', 'tqdm', 'matplotlib'],
    keywords=['python', 'neural networks', 'deep learning', 'predictive coding'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)