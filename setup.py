from setuptools import setup
from os import path
import os
import subprocess
import sys

# https://stackoverflow.com/a/65622116 ¯\_(ツ)_/¯
if sys.platform in ["win32", "cygwin"]:
    os.environ["DISTUTILS_USE_SDK"] = "1"


def build_native(spec):
    c_api_dir = path.abspath(path.join(path.dirname(__file__), "c-api"))

    if sys.platform == "darwin":
        # Build universal2 fat binary for both x86_64 and arm64
        subprocess.run(
            ["rustup", "target", "add", "x86_64-apple-darwin", "aarch64-apple-darwin"],
            check=True,
            cwd=c_api_dir,
        )

        # Build for x86_64
        subprocess.run(
            ["cargo", "build", "--release", "--target", "x86_64-apple-darwin"],
            check=True,
            cwd=c_api_dir,
        )

        # Build for arm64
        subprocess.run(
            ["cargo", "build", "--release", "--target", "aarch64-apple-darwin"],
            check=True,
            cwd=c_api_dir,
        )

        # Combine with lipo into a fat binary
        lib_x86 = path.join(
            c_api_dir, "target", "x86_64-apple-darwin", "release", "libomikuji.dylib"
        )
        lib_arm = path.join(
            c_api_dir, "target", "aarch64-apple-darwin", "release", "libomikuji.dylib"
        )
        fat_lib = path.join(c_api_dir, "target", "release", "libomikuji.dylib")

        subprocess.run(
            ["lipo", "-create", "-output", fat_lib, lib_x86, lib_arm],
            check=True,
        )

        # Use a no-op command for milksnake since we've already built everything
        build_cmd = ["echo", "universal2 fat binary already built"]
    else:
        build_cmd = ["cargo", "build", "--release"]

    build = spec.add_external_build(cmd=build_cmd, path="c-api")
    spec.add_cffi_module(
        module_path="omikuji._libomikuji",
        dylib=lambda: build.find_dylib("omikuji", in_path="target/release"),
        header_filename=lambda: build.find_header(
            "omikuji.h", in_path="target/include"
        ),
        rtld_flags=["NOW", "NODELETE"],
    )


def load_readme():
    curr_dir = path.abspath(path.dirname(__file__))
    with open(path.join(curr_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


setup(
    name="omikuji2",
    version="0.5.1",
    author="Tom Dong",
    author_email="tom.tung.dyb@gmail.com",
    description=(
        "Python binding to Omikuji, an efficient implementation of Partioned Label "
        "Trees and its variations for extreme multi-label classification"
    ),
    long_description=load_readme(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    url="https://github.com/tomtung/omikuji",
    license="MIT",
    packages=["omikuji2"],
    package_dir={"": "python-wrapper"},
    zip_safe=False,
    platforms="any",
    setup_requires=["milksnake>=0.1.6"],
    install_requires=["milksnake>=0.1.6"],
    milksnake_tasks=[build_native],
    milksnake_universal=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Rust",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
)
