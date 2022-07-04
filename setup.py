#! /usr/bin/env python
"""The Word Embeddings Fairness Evaluation Framework"""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join("wefe", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "wefe"
DESCRIPTION = "The Word Embedding Fairness Evaluation Framework"
with codecs.open("README.rst", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = "WEFE Team"
MAINTAINER = "WEFE Team"
MAINTAINER_EMAIL = "pablo.badilla@ug.uchile.cl"
URL = "https://github.com/dccuchile/wefe"
LICENSE = "new BSD"
DOWNLOAD_URL = "https://github.com/dccuchile/wefe"
VERSION = __version__
INSTALL_REQUIRES = [
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "gensim",
    "plotly",
    "six",
    "requests",
    "semantic_version",
    "tqdm",
]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]
EXTRAS_REQUIRE = {
    "tests": ["pytest", "pytest-cov"],
    "docs": ["sphinx", "sphinx-gallery", "sphinx_rtd_theme", "numpydoc", "matplotlib"],
}

setup(
    name=DISTNAME,
    author=AUTHOR,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.6",
    include_package_data=True,
)

