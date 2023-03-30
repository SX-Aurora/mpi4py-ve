set -x
mpiexec -veo -np 2 ./ping-pong-c --dev1=vh --dev1_node=0 --dev2=vh --dev2_node=0 --n=32 --loop_count=10
mpiexec -veo -np 2 ./ping-pong-c --dev1=vh --dev1_node=0 --dev2=ve --dev2_node=1 --n=32 --loop_count=10
mpiexec -veo -np 2 ./ping-pong-c --dev1=ve --dev1_node=1 --dev2=vh --dev2_node=0 --n=32 --loop_count=10
VE_OMP_NUM_THREADS=1 mpiexec -veo -np 2 ./ping-pong-c --dev1=ve --dev1_node=1 --dev2=ve --dev2_node=1 --n=32 --loop_count=10
mpiexec -veo -np 2 ./ping-pong-c --dev1=ve --dev1_node=1 --dev2=ve --dev2_node=2 --n=32 --loop_count=10
set +x
