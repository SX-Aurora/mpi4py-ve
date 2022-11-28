.PHONY: test

PYTHON=python

M=10000
N=10000
DTYPE=float
ITER=100

test:
	echo "GEMV on  VH ( 1 process)" &&                           mpirun -veo -np 1  ${PYTHON} gemv.py -dev vh -dtype ${DTYPE} -m ${M} -n ${N} -iter ${ITER}
	echo "GEMV on  VH ( 2 process)" &&                           mpirun -veo -np 2  ${PYTHON} gemv.py -dev vh -dtype ${DTYPE} -m ${M} -n ${N} -iter ${ITER}
	echo "GEMV on  VH ( 4 process)" &&                           mpirun -veo -np 4  ${PYTHON} gemv.py -dev vh -dtype ${DTYPE} -m ${M} -n ${N} -iter ${ITER}
	echo "GEMV on 1VE ( 1 process)" && VE_NLCPY_NODELIST=0       mpirun -veo -np 1  ${PYTHON} gemv.py -dev ve -dtype ${DTYPE} -m ${M} -n ${N} -iter ${ITER}
	echo "GEMV on 2VE ( 2 process)" && VE_NLCPY_NODELIST=0,1     mpirun -veo -np 2  ${PYTHON} gemv.py -dev ve -dtype ${DTYPE} -m ${M} -n ${N} -iter ${ITER}
	echo "GEMV on 4VE ( 4 process)" && VE_NLCPY_NODELIST=0,1,2,3 mpirun -veo -np 4  ${PYTHON} gemv.py -dev ve -dtype ${DTYPE} -m ${M} -n ${N} -iter ${ITER}
	echo "GEMV on 4VE (32 process)" && VE_NLCPY_NODELIST=0,1,2,3 mpirun -veo -np 32 ${PYTHON} gemv.py -dev ve -dtype ${DTYPE} -m ${M} -n ${N} -iter ${ITER}
