module purge

# Compilers
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

#---go to the work directory
cd $WORK
#---clear working directory
make clobber
#---compile adcirc, aswip, adcprep, and parallel adcirc
make adcirc aswip adcprep padcirc compiler=gfortran NETCDF=enable NETCDFHOME=$DIR NETCDF4=enable NETCDF4_COMPRESSION=enable
