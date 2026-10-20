.PHONY: test

PYTHON=python

test:
	echo "Thermal Simulation on  VH (8 process)" &&                           mpirun-py -veo -np 8 ${PYTHON} thermal.py -dev vh
	echo "Thermal Simulation on 4VE (4 process)" && VE_NLCPY_NODELIST=0,1,2,3 mpirun-py -veo -np 4 ${PYTHON} thermal.py -dev ve
