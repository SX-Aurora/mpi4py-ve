# Copyright(C) 2026 NEC Corporation. All Rights Reserved.
#
# NEC Numeric Library Collection

set MPIRUN_WRAPPER=mpirun-py
set ARGS=($_)
set SELF_PATH=$ARGS[2]
set FULL_PATH=`dirname ${SELF_PATH}`
set SELF_DIR=`realpath ${FULL_PATH}`

if ( -x "${SELF_DIR}/${MPIRUN_WRAPPER}" ) then
    setenv PATH ${SELF_DIR}/:${PATH}
else
    echo "Error: ${MPIRUN_WRAPPER} command is not found. Please check if mpi4py-ve is installed correctly."
endif

unset MPIRUN_WRAPPER
unset ARGS
unset SELF_PATH
unset FULL_PATH
unset SELF_DIR
