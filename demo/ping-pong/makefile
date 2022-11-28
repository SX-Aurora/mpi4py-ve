.PHONY: test

PYTHON=python

N=25
ITER=10

test:
	echo "Ping-Pong VH-VH" &&                                     mpirun -veo -np 2 python ping-pong.py -dev1 vh -dev2 vh -n ${N} -loop_count ${ITER}
	echo "Ping-Pong VH-VE" &&                                     mpirun -veo -np 2 python ping-pong.py -dev1 vh -dev2 ve -n ${N} -loop_count ${ITER}
	echo "Ping-Pong VE-VE(same      VE)" &&                       mpirun -veo -np 2 python ping-pong.py -dev1 ve -dev2 ve -n ${N} -loop_count ${ITER}
	echo "Ping-Pong VE-VE(different VE)" && VE_NLCPY_NODELIST=0,1 mpirun -veo -np 2 python ping-pong.py -dev1 ve -dev2 ve -n ${N} -loop_count ${ITER}
