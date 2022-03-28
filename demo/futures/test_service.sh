#!/bin/bash

PYTHON=${1-${PYTHON-python}}
MPIEXEC=${MPIEXEC-mpiexec}
testdir=$(dirname "$0")

set -e

if [ $(command -v mpichversion) ]; then
    $MPIEXEC -n 1 $PYTHON -m mpi4pyve.futures.server --xyz > /dev/null 2>&1 || true
    $MPIEXEC -n 2 $PYTHON -m mpi4pyve.futures.server --bind localhost &
    mpi4pyveserver=$!; sleep 0.25;
    $MPIEXEC -n 1 $PYTHON $testdir/test_service.py --host localhost
    wait $mpi4pyveserver
    $MPIEXEC -n 2 $PYTHON -m mpi4pyve.futures.server --port 31414 --info "a=x,b=y" &
    mpi4pyveserver=$!; sleep 0.25;
    $MPIEXEC -n 1 $PYTHON $testdir/test_service.py --port 31414 --info "a=x,b=y"
    wait $mpi4pyveserver
fi

if [ $(command -v mpichversion) ] && [ $(command -v hydra_nameserver) ]; then
    hydra_nameserver &
    nameserver=$!; sleep 0.25;
    $MPIEXEC -nameserver localhost -n 2 $PYTHON -m mpi4pyve.futures.server &
    mpi4pyveserver=$!; sleep 0.25;
    $MPIEXEC -nameserver localhost -n 1 $PYTHON $testdir/test_service.py
    wait $mpi4pyveserver
    $MPIEXEC -nameserver localhost -n 2 $PYTHON -m mpi4pyve.futures.server --service test-service &
    mpi4pyveserver=$!; sleep 0.25;
    $MPIEXEC -nameserver localhost -n 1 $PYTHON $testdir/test_service.py --service test-service
    wait $mpi4pyveserver
    kill -TERM $nameserver
    wait $nameserver 2>/dev/null || true
fi
