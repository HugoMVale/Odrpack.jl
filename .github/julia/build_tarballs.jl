using BinaryBuilder

name = "odrpack95"
version = v"2.0.1"

# Collection of sources required to build ECOSBuilder
sources = [
    GitSource("https://github.com/HugoMVale/odrpack95.git", "e440c9dc220a6d74e251181c8eb9e8fd430a64b4")
]

platforms = [
    Platform("x86_64", "linux"; libc="glibc")
]

products = [
    LibraryProduct("libodrpack95", :libodrpack)
]

dependencies = [
    Dependency("OpenBLAS_jll")
]

script = raw"""
cd $WORKSPACE/srcdir/odrpack95

# Set pkg-config path so Meson can find openblas.pc
export PKG_CONFIG_PATH="${prefix}/lib/pkgconfig"

# Optional: show what pkg-config sees
pkg-config --list-all | grep blas || true

# Configure Meson
meson setup builddir --prefix=$prefix --libdir=lib -Dbuild_shared=true

# Build and install
meson compile -C builddir -j${nproc}
meson install -C builddir
"""

build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
    julia_compat="1.6")