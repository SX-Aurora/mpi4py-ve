# Copyright(C) 2026 NEC Corporation. All Rights Reserved.
#
# NEC Numeric Library Collection

MPIRUN_WRAPPER=mpirun-py
SELF_PATH=${BASH_SOURCE:-$0}
SELF_DIR=$(realpath $(dirname ${SELF_PATH}))

if [ -x "${SELF_DIR}/${MPIRUN_WRAPPER}" ]; then
    export PATH=${SELF_DIR}/:${PATH}
else
    echo "Error: ${MPIRUN_WRAPPER} command is not found. Please check if mpi4py-ve is installed correctly."
fi

unset MPIRUN_WRAPPER
unset SELF_PATH
unset SELF_DIR
