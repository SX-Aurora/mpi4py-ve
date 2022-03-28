#!/bin/bash

MPIEXEC=${MPIEXEC-mpiexec}
PYTHON=${1-${PYTHON-python}}
export PYTHONDONTWRITEBYTECODE=1

$PYTHON -m coverage erase

$MPIEXEC -n 1 $PYTHON -m coverage run "$(dirname "$0")/coverage-helper.py" > /dev/null || true

$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.bench --help > /dev/null
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.bench --threads             helloworld -q
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.bench --no-threads          helloworld -q
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.bench --thread-level=single helloworld -q
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.bench helloworld > /dev/null
$MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.bench helloworld > /dev/null
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.bench helloworld > /dev/null
$MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.bench helloworld -q
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.bench ringtest > /dev/null
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.bench ringtest -q -l 2 -s 1
$MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.bench ringtest -q -l 2 -s 1
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.bench              > /dev/null 2>&1 || true
$MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.bench              > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.bench qwerty       > /dev/null 2>&1 || true
$MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.bench qwerty       > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.bench --mpe qwerty > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.bench --vt  qwerty > /dev/null 2>&1 || true

$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.run --help > /dev/null
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve --version > /dev/null
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve --help > /dev/null
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve - < /dev/null
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -c "42" > /dev/null
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -m this > /dev/null
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve "$(dirname "$0")/coverage-helper.py" > /dev/null || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -rc threads=0 --rc=thread_level=single -c "" > /dev/null
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -p mpe -profile mpe -c ""          > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve --profile mpe --profile=mpe -c ""  > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -vt --vt -mpe --mpe -c ""          > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve                                    > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -m                                 > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -c                                 > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -p                                 > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -bad                               > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve --bad=a                            > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -rc=                               > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve --rc=a                             > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve --rc=a=                            > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve --rc==a                            > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -c "import sys; sys.exit()"        > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -c "import sys; sys.exit(0)"       > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -c "import sys; sys.exit(1)"       > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -c "import sys; sys.exit('error')" > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve -c "from mpi4pyve import MPI; 1/0;"  > /dev/null 2>&1 || true

$MPIEXEC -n 1 $PYTHON -m coverage run demo/futures/test_futures.py -q 2> /dev/null
$MPIEXEC -n 2 $PYTHON -m coverage run demo/futures/test_futures.py -q 2> /dev/null
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.futures demo/futures/test_futures.py -q 2> /dev/null
$MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.futures demo/futures/test_futures.py -q ProcessPoolPickleTest 2> /dev/null
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.futures -h > /dev/null
$MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.futures -h > /dev/null
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.futures -m this > /dev/null
$MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.futures -m this > /dev/null
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.futures -c "42"
$MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.futures -c "42"
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.futures - </dev/null
$MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.futures - </dev/null
$MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.futures -c "raise SystemExit"
$MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.futures -c "raise SystemExit()"
$MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.futures -c "raise SystemExit(0)"
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.futures                           > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.futures xy                        > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.futures -c                        > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.futures -m                        > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.futures -x                        > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.futures -c "1/0"                  > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.futures -c "raise SystemExit(11)" > /dev/null 2>&1 || true
$MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.futures -c "raise SystemExit('')" > /dev/null 2>&1 || true
if [ $(command -v mpichversion) ]; then
    testdir=demo/futures
    $MPIEXEC -n 1 $PYTHON -m coverage run -m mpi4pyve.futures.server --xyz > /dev/null 2>&1 || true
    $MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.futures.server --bind localhost &
    mpi4pyveserver=$!; sleep 1;
    $MPIEXEC -n 1 $PYTHON -m coverage run $testdir/test_service.py --host localhost
    wait $mpi4pyveserver
    $MPIEXEC -n 2 $PYTHON -m coverage run -m mpi4pyve.futures.server --port 31414 --info "a=x,b=y" &
    mpi4pyveserver=$!; sleep 1;
    $MPIEXEC -n 1 $PYTHON -m coverage run $testdir/test_service.py --port 31414 --info "a=x,b=y"
    wait $mpi4pyveserver
fi
if [ $(command -v mpichversion) ] && [ $(command -v hydra_nameserver) ]; then
    testdir=demo/futures
    hydra_nameserver &
    nameserver=$!; sleep 1;
    $MPIEXEC -nameserver localhost -n 2 $PYTHON -m coverage run -m mpi4pyve.futures.server &
    mpi4pyveserver=$!; sleep 1;
    $MPIEXEC -nameserver localhost -n 1 $PYTHON -m coverage run $testdir/test_service.py
    wait $mpi4pyveserver
    kill -TERM $nameserver
    wait $nameserver 2>/dev/null || true
fi

$PYTHON -m coverage combine
