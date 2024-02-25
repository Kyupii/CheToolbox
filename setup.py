from setuptools import setup, find_packages

VERSION = '0.0.9'
DESCRIPTION = 'A library of common chemical engineering calculations'
LONG_DESCRIPTION = 'A personal project cataloging common chemical engineering calculations employing numerical methods'

setup(
    name = 'chetoolbox',
    version = VERSION,
    author = 'Quan Phan & Ethan Molnar',
    author_email = 'Phanqv@mail.uc.edu',
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    packages = find_packages(),
    install_requires = ['numpy',
                        'scipy',
                        ],

    keywords = ["python", "chemical engineering"],
    classifiers = ["Development Status :: 3 - Alpha", # Either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
                   "Intended Audience :: Education",
                   "Programming Language :: Python :: 2",
                   "Programming Language :: Python :: 3",
                   "Operating System :: Microsoft :: Windows", 
                ]
)