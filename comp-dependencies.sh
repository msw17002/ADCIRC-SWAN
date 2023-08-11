#!/bin/bash

#===Parameters
echo "==============================================================================="
echo "=======================EXPORTING PARAMETERS===================================="
echo 'export HOME=$(pwd)'
echo 'export DIR=${HOME}/apps/Library'
echo "export CC=gcc"
echo "export CXX=g++"
echo "export FC=gfortran"
echo "export F77=gfortran"
echo 'export apps=${HOME}/apps'
echo "export ZLIB=zlib-1.2.13"
echo "export HDF5=hdf5-1.12.2"
echo "export NTCC=netcdf-c-4.9.0"
echo "export NTCF=netcdf-fortran-4.6.0"
echo "export MPIC=mpich-4.0.3"
echo "==============================================================================="
sleep 5
export HOME=$(pwd)
export DIR=${HOME}/apps/Library
export CC=gcc
export CXX=g++
export FC=gfortran
export F77=gfortran
export apps=${HOME}/apps
export ZLIB=zlib-1.2.13
export HDF5=hdf5-1.12.2
export NTCC=netcdf-c-4.9.0
export NTCF=netcdf-fortran-4.6.0
export MPIC=mpich-4.0.3
export DT=5

echo "==============================================================================="
echo "===================CREATING DIRECTORY STRUCTURE================================"
sleep 5
#===Set directory structure
#-  apps directory
if [ -d "$HOME/apps" ]; then 
   echo "$HOME/apps already exists!"
else
   echo 'mkdir $HOME/apps'
   sleep $DT 
   mkdir $HOME/apps
fi 
#-  Downloads directory
if [ -d "$HOME/apps/Downloads" ]; then
   echo "$HOME/apps/Downloads already exists!"
else
   echo 'mkdir $HOME/apps/Downloads'
   sleep $DT
   mkdir $HOME/apps/Downloads
fi
#-  Library directory
if [ -d "$HOME/apps/Library" ]; then
   echo "$HOME/apps/Library already exists!"
else
   echo 'mkdir $HOME/apps/Library'
   sleep $DT
   mkdir $HOME/apps/Library
fi
echo "==============================================================================="

echo "==============================================================================="
echo "===================DOWNLOADING SOURCE BINARIES ================================"
sleep 5
#===conditionally download dependancies
#-  zlib source code
if [ -f "$HOME/apps/Downloads/$ZLIB.tar.gz" ]; then
    echo "$HOME/apps/Downloads/$ZLIB.tar.gz already exists!"
else
    echo 'wget -q 'https://github.com/madler/zlib/releases/download/v1.2.13/zlib-1.2.13.tar.gz' -O $HOME/apps/Downloads/$ZLIB.tar.gz'
    sleep $DT
    wget -q 'https://github.com/madler/zlib/releases/download/v1.2.13/zlib-1.2.13.tar.gz' -O $HOME/apps/Downloads/$ZLIB.tar.gz
fi
#-  HDF5 source code
if [ -f "$HOME/apps/Downloads/$HDF5.tar.gz" ]; then
    echo "$HOME/apps/Downloads/$HDF5.tar.gz already exists!"
else
    echo "wget -q 'https://www.hdfgroup.org/package/hdf5-1-12-2-tar-gz/?wpdmdl=16474&amp;refresh=64d28450662021691518032' -O "'$HOME/apps/Downloads/$HDF5.tar.gz'
    sleep $DT
    wget -q 'https://www.hdfgroup.org/package/hdf5-1-12-2-tar-gz/?wpdmdl=16474&amp;refresh=64d28450662021691518032' -O $HOME/apps/Downloads/$HDF5.tar.gz
fi
#-  netcdf4-C source code
if [ -f "$HOME/apps/Downloads/$NTCC.tar.gz" ]; then
    echo "$HOME/apps/Downloads/$NTCC.tar.gz already exists!"
else
    echo 'wget -q 'https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.9.0.tar.gz' -O $HOME/apps/Downloads/$NTCC.tar.gz'
    sleep $DT
    wget -q 'https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.9.0.tar.gz' -O $HOME/apps/Downloads/$NTCC.tar.gz
fi
#-  netcdf4-gfortran source code
if [ -f "$HOME/apps/Downloads/$NTCF.tar.gz" ]; then
    echo "$HOME/apps/Downloads/$NTCF.tar.gz already exists!"
else
    echo 'wget -q 'https://downloads.unidata.ucar.edu/netcdf-fortran/4.6.0/netcdf-fortran-4.6.0.tar.gz' -O $HOME/apps/Downloads/$NTCF.tar.gz'
    sleep $DT
    wget -q 'https://downloads.unidata.ucar.edu/netcdf-fortran/4.6.0/netcdf-fortran-4.6.0.tar.gz' -O $HOME/apps/Downloads/$NTCF.tar.gz
fi
#-  mpich source code
if [ -f "$HOME/apps/Downloads/$MPIC.tar.gz" ]; then 
    echo "$HOME/apps/Downloads/$MPIC.tar.gz already exists!"
else
    echo 'wget -q 'https://www.mpich.org/static/downloads/4.0.3/mpich-4.0.3.tar.gz' -O $HOME/apps/Downloads/$NTCF.tar.gz'
    sleep $DT
    wget -q 'https://www.mpich.org/static/downloads/4.0.3/mpich-4.0.3.tar.gz' -O $HOME/apps/Downloads/$MPIC.tar.gz
fi
echo "==============================================================================="

echo "==============================================================================="
echo "==========================COMPILING ZLIB======================================="
echo 'cd $HOME/apps/Downloads'
echo 'tar -xvzf ${ZLIB}.tar.gz'
echo 'cd ${ZLIB}/'
echo './configure --prefix=$DIR'
echo "make"
echo "make install"
echo "==============================================================================="
sleep 5
#===compile zlib
cd $HOME/apps/Downloads
tar -xzf ${ZLIB}.tar.gz
cd ${ZLIB}/
./configure --prefix=$DIR
make
make install
echo "==============================================================================="
echo "==========================COMPILING HDF5======================================="
echo 'cd $HOME/apps/Downloads'
echo 'tar -xvzf ${HDF5}.tar.gz'
echo 'cd ${HDF5}'
echo './configure --prefix=$DIR --with-zlib=$DIR --enable-hl --enable-fortran'
echo "make"
echo "make install"
echo "==============================================================================="
sleep 5
#===compile hdf5 library for netcdf4 functionality
cd $HOME/apps/Downloads
tar -xzf ${HDF5}.tar.gz
cd ${HDF5}
./configure --prefix=$DIR --with-zlib=$DIR --enable-hl --enable-fortran
make
make install
export HDF5=$DIR
export LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH
echo "==============================================================================="
echo "========================COMPILING NETCDF4-C===================================="
echo 'export HDF5=$DIR'
echo 'export LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH'
echo 'cd $HOME/apps/Downloads'
echo 'tar -xvzf ${NTCC}.tar.gz'
echo 'cd ${NTCC}/'
echo 'export CPPFLAGS=-I$DIR/include'
echo 'export LDFLAGS=-L$DIR/lib'
echo './configure --prefix=$DIR --disable-dap'
echo "make"
echo "make install"
echo "==============================================================================="
sleep 5
#===compile netcdf4-C library
cd $HOME/apps/Downloads
tar -xzf ${NTCC}.tar.gz
cd ${NTCC}/
export CPPFLAGS=-I$DIR/include 
export LDFLAGS=-L$DIR/lib
./configure --prefix=$DIR --disable-dap
make
make install
export PATH=$DIR/bin:$PATH
export NETCDF=$DIR
echo "==============================================================================="
echo "========================COMPILING NETCDF4-fortran=============================="
echo 'export PATH=$DIR/bin:$PATH'
echo 'export NETCDF=$DIR'
echo 'cd $HOME/apps/Downloads'
echo 'tar -xvzf ${NTCF}.tar.gz'
echo 'cd ${NTCF}/'
echo 'export LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH'
echo 'export CPPFLAGS=-I$DIR/include'
echo 'export LDFLAGS=-L$DIR/lib'
echo 'export LIBS="-lnetcdf -lhdf5_hl -lhdf5 -lz"'
echo './configure --prefix=$DIR --disable-shared'
echo "make check"
echo "make"
echo "make install"
echo "==============================================================================="
sleep 5
#===compile netcdf4-gfortran library
cd $HOME/apps/Downloads
tar -xzf ${NTCF}.tar.gz
cd ${NTCF}/
export LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH
export CPPFLAGS=-I$DIR/include 
export LDFLAGS=-L$DIR/lib
export LIBS="-lnetcdf -lhdf5_hl -lhdf5 -lz" 
./configure --prefix=$DIR --disable-shared
make check
make
make install
#===compile mpich for parallelization 
echo "==============================================================================="
echo "========================COMPILING MPICH=============================="
echo "cd $HOME/apps/Downloads"
echo "cd ${MPIC}/"
echo "./configure --prefix=$DIR"
echo "make"
echo "make install"
echo "==============================================================================="
sleep 5
cd $HOME/apps/Downloads
tar -xvzf ${MPIC}.tar.gz
cd ${MPIC}/
./configure --prefix=$DIR
make
make install
