#!/bin/sh
python -m cython --cleanup 3 -w src $@ mpi4pyve.MPI.pyx && \
mv src/mpi4pyve.MPI*.h src/mpi4pyve/include/mpi4pyve
