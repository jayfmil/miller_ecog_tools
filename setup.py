
"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

requirements = []
setup_requirements = []

setup(
    name='RAM_ECoG',
    version=0.1,
    description="A framework for doing ECoG analyses of RAM data",
    long_description=long_description,
    author="Jonathan Miller",
    url='https://github.com/jayfmil/RAM_ECoG',
    packages=find_packages(exclude=["Projects"]),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='RAM_ECoG',
    setup_requires=setup_requirements,
)