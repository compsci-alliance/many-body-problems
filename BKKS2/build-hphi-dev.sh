#!/bin/sh
set -x

# コードの展開とパッチの適用
#tar zxf qlmpack-develop.tar.gz
cd qlmpack-develop
patch -p1 < ../HPhi-dev.patch

# cmake & build & install
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/HPhi -DCMAKE_C_COMPILER=gcc-7 -DCMAKE_Fortran_COMPILER=gfortran-7 ..
make
make install

cd ..
mkdir -p $HOME/HPhi/share/hphi
cp -rp samples $HOME/HPhi/share/hphi
