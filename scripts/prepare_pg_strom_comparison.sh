#!/usr/bin/env bash
# Build the pinned PG-Strom release against a private PostgreSQL installation.
set -euo pipefail
runtime=${1:?Pass a new private runtime directory}
pg_prefix=${PG_STROM_PG_PREFIX:-/home/ubuntu/pg18}
cuda_prefix=${PG_STROM_CUDA_PREFIX:-/home/ubuntu/cuda-12.6}
cuda_driver_lib=${PG_STROM_CUDA_DRIVER_LIB:-/opt/thunder/nvidia/lib}
mkdir "$runtime"
cd "$runtime"
runtime=$(pwd)
git clone --depth 1 --branch v6.1 https://github.com/heterodb/pg-strom.git source
test "$(git -C source rev-parse HEAD)" = 4d12ef415759dc48cd4c1421565e9c694b7bd3f9
cp -a "$pg_prefix" pg18
cp -a "$cuda_prefix" cuda

nvidia_redistribution=https://developer.download.nvidia.com/compute/cuda/redist
cufile=libcufile-linux-x86_64-1.11.1.6-archive
profiler=cuda_profiler_api-linux-x86_64-12.6.77-archive
objdump=cuda_cuobjdump-linux-x86_64-12.6.77-archive
curl -fsSL "$nvidia_redistribution/libcufile/linux-x86_64/$cufile.tar.xz" -o "$cufile.tar.xz"
curl -fsSL "$nvidia_redistribution/cuda_profiler_api/linux-x86_64/$profiler.tar.xz" -o "$profiler.tar.xz"
curl -fsSL "$nvidia_redistribution/cuda_cuobjdump/linux-x86_64/$objdump.tar.xz" -o "$objdump.tar.xz"
printf '%s\n' \
    "99ee42e0ebf826f13cea9d15d63f54f225ddc42ff87be1427c58be9692ae25bc  $cufile.tar.xz" \
    "0807ec2335ea358a96f984ce08cc0b3d572b5cc486f9082fc3afa9a1773f2a58  $profiler.tar.xz" \
    "54d0edf14284249baf8e2db1c26c71d079ad7afd1f41779e22f239ff9e8bf60f  $objdump.tar.xz" \
    | sha256sum -c
tar xf "$cufile.tar.xz"
tar xf "$profiler.tar.xz"
tar xf "$objdump.tar.xz"
cp -a "$cufile/include/." "$profiler/include/." cuda/include/
cp -a "$cufile/lib/." cuda/lib/
cp -a "$objdump/bin/cuobjdump" cuda/bin/

python3 -m pip install --no-deps --target arrow-wheel pyarrow==21.0.0
mkdir pc bin socket
ln -s "$(command -v pkg-config)" bin/pkgconf
export PATH="$runtime/bin:$PATH"
export CPLUS_INCLUDE_PATH="$runtime/arrow-wheel/pyarrow/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export PKG_CONFIG_PATH="$runtime/pc${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
for library in arrow parquet; do
    ln -s "lib$library.so.2100" "arrow-wheel/pyarrow/lib$library.so"
    cat > "pc/$library.pc" <<PC
prefix=$runtime/arrow-wheel/pyarrow
libdir=\${prefix}
includedir=\${prefix}/include
Name: $library
Description: Apache $library C++ libraries from the PyArrow 21.0.0 wheel
Version: 21.0.0
Libs: -L\${libdir} -l$library -larrow
Cflags: -I\${includedir}
PC
done

make_options=(PG_CONFIG="$runtime/pg18/bin/pg_config" CUDA_PATH="$runtime/cuda" CUDA_LPATH="$cuda_driver_lib")
make -C source/src -j4 "${make_options[@]}"
make -C source/src "${make_options[@]}" install
pg18/bin/initdb --no-locale -D db
cat >> db/postgresql.conf <<CONF
listen_addresses = ''
port = 55436
unix_socket_directories = '$runtime/socket'
shared_preload_libraries = 'pg_strom'
pg_strom.gpudirect_enabled = off
CONF
