.PHONY: test

MPIEXEC=mpiexec -veo -n 1
PYTHON=python

test:
	echo 100 | ${MPIEXEC} ${PYTHON} cpi-cco.py
	echo 100 | ${MPIEXEC} ${PYTHON} cpi-rma.py
