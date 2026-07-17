from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

SRC = Path(__file__).resolve().parent.parent / "src"

ext_modules = [
    Pybind11Extension(
        "vptree_mnist",
        [str(SRC / "vptree_py.cpp")],
        include_dirs=[str(SRC)],
        cxx_std=17,
        extra_compile_args=["-O3", "-Wall"],
    ),
]

setup(
    name="vptree_mnist",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
