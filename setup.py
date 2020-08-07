#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()


requirements = [ ]

setup_requirements = [ ]


setup(
    author="Jenisha Patel",
    author_email='jenisha_p@hotmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Code to process raw scanned ionogram data from the Alouette-I satellite",
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords='alouette_scanned_ionograms_processing',
    name='alouette_scanned_ionograms_processing',
    packages=find_packages(include=['scan2data']),
    setup_requires=setup_requirements,
    version='0.1.0',
)
