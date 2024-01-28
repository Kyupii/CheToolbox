from setuptools import setup, find_packages

VERSION = '1.1'
DESCRIPTION = 'A library of commoon chemical engineering calculations'
LONG_DESCRIPTION = 'A personal project cataloging common chemical engineering calculations employing numerical methods'

setup(
    name = 'CheToolbox',
    version = VERSION,
    author = 'Quan Phan & Ethan Molnar',
    author_email = 'Phanqv@mail.uc.edu',
    description= DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages = find_packages(),
    install_requires = ['numpy',
                     'pandas'],
    
    keywords = ["python", "Chemical Engineering"],
    classifiers= [
    "Intended Audience :: Education",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Operating System :: Microsoft :: Windows",
    ]


)