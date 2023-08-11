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
echo "=========================COMPILING ADCIRC======================================"
echo 'cd $WORK'
echo 'make clobber'
echo 'make adcirc aswip adcprep padcirc compiler=gfortran NETCDF=enable NETCDFHOME=$DIR NETCDF4=enable NETCDF4_COMPRESSION=enable'
echo "==============================================================================="
sleep 5

#---go to the work directory
cd $WORK
#---clear working directory
make clobber
#---compile adcirc, aswip, adcprep, and parallel adcirc
make adcirc aswip adcprep padcirc compiler=gfortran NETCDF=enable NETCDFHOME=$DIR NETCDF4=enable NETCDF4_COMPRESSION=enable
