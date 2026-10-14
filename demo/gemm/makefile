.PHONY: test

PYTHON=python

N=10000
DTYPE=float

test:
	echo "GEMM on  VH (4 process)" &&                           mpirun-py -veo -np 4 ${PYTHON} gemm.py -dev vh -dtype ${DTYPE} -n ${N}
	echo "GEMM on 1VE (1 process)" && VE_NLCPY_NODELIST=0       mpirun-py -veo -np 1 ${PYTHON} gemm.py -dev ve -dtype ${DTYPE} -n ${N}
	echo "GEMM on 4VE (4 process)" && VE_NLCPY_NODELIST=0,1,2,3 mpirun-py -veo -np 4 ${PYTHON} gemm.py -dev ve -dtype ${DTYPE} -n ${N}
