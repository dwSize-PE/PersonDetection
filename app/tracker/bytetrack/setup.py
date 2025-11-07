from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name="cython_bbox",
    version="0.1",
    ext_modules=cythonize([
        Extension(
            "cython_bbox",
            sources=["cython_bbox.pyx"],
            include_dirs=[np.get_include()],
        )
    ]),
)
