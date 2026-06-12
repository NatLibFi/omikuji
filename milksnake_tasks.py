from milksnake.api import Specification
from os import path
import sys


def build_native(spec):
    c_api_dir = path.abspath(path.join(path.dirname(__file__), "c-api"))

    if sys.platform == "darwin":
        # Build universal2 fat binary for both x86_64 and arm64
        import subprocess

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
