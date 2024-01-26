echo "==============================================================================="
echo "=======================EXPORTING PARAMETERS===================================="
echo 'module purge'
echo 'export HOME=$(pwd)'
echo 'export DIR=${HOME}/apps/Library'
echo 'export CC=gcc'
echo 'export CXX=g++'
echo 'export FC=gfortran'
echo 'export F77=gfortran'
echo 'export LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH'
echo 'export CPPFLAGS=-I$DIR/include'
echo 'export LDFLAGS=-L$DIR/lib'
echo 'export LIBS="-lnetcdf -lhdf5_hl -lhdf5 -lz"'
echo 'export PATH=${DIR}/bin:$PATH'
echo 'export FFLAGS=-fallow-argument-mismatch'
echo 'export FCFLAGS=-fallow-argument-mismatch'
echo 'export WORK=/home/data/ADCIRC/source-code/work'
echo "==============================================================================="
sleep 5

module purge
export HOME=$(pwd)
export DIR=${HOME}/apps/Library
export CC=gcc
export CXX=g++
export FC=gfortran
export F77=gfortran
export LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH
export CPPFLAGS=-I$DIR/include
export LDFLAGS=-L$DIR/lib
export LIBS="-lnetcdf -lhdf5_hl -lhdf5 -lz"
export PATH=${DIR}/bin:$PATH
export FFLAGS=-fallow-argument-mismatch
export FCFLAGS=-fallow-argument-mismatch
export WORK=/home/data/ADCIRC/source-code/work

echo "==============================================================================="
echo "======================COMPILE MODELING SYSTEM=================================="
echo 'git clone https://github.com/adcirc/adcirc.git'
echo 'cd ./adcirc'
echo 'mkdir build'
echo 'cd ./build'
echo 'cmake .. -DBUILD_ADCIRC=ON \
         -DBUILD_PADCIRC=ON \
         -DBUILD_ADCSWAN=ON \
         -DBUILD_PADCSWAN=ON \
         -DBUILD_ADCPREP=ON \
         -DBUILD_UTILITIES=ON \
         -DBUILD_ASWIP=ON \
         -DBUILD_SWAN=ON \
         -DBUILD_PUNSWAN=ON \
         -DENABLE_OUTPUT_NETCDF=ON \
         -DNETCDFHOME=${HOME}/apps/Library'

#---clone the ADCIRC github repository
git clone https://github.com/adcirc/adcirc.git

#---navigate and create directory structure
cd ./adcirc
mkdir build 
cd ./build 

#---compile utilzing cmake
cmake .. -DBUILD_ADCIRC=ON \
         -DBUILD_PADCIRC=ON \
         -DBUILD_ADCSWAN=ON \
         -DBUILD_PADCSWAN=ON \
         -DBUILD_ADCPREP=ON \
         -DBUILD_UTILITIES=ON \
         -DBUILD_ASWIP=ON \
         -DBUILD_SWAN=ON \
         -DBUILD_PUNSWAN=ON \
         -DENABLE_OUTPUT_NETCDF=ON \
         -DNETCDFHOME=${HOME}/apps/Library
