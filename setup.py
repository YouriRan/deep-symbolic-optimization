from setuptools import setup
import os
from setuptools import dist


required = [
    "numpy",
    "pytorch",
    "lightning"
]
    
setup(  name='dso',
        version='1.0dev',
        description='Deep symbolic optimization.',
        author='YR',
        packages=['dso'],
        install_requires=required,
        )
