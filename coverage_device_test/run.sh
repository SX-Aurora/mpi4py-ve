#!/bin/bash

TEST='full'
NP=4
PYTHON_CMD=python
MPIRUN_CMD=mpirun
TEST_DEVICE='ve_vh'
HOSTS=''
HOSTS_OPT=''

function usage() {
  echo 'Usage: run.sh [ARGUMENT]...'
  echo ''
  echo '  ARGUMENT:'
  echo '  --test MODE or --test=MODE: specify the test MODE'
  echo '         available MODE are [full|small]'
  echo '         (default: full)'
  echo '  --np NO or --np=NO: specify the total number of processes'
  echo '         (default: 4)'
  echo '  --python-cmd CMD or --python-cmd=CMD: specify python command'
  echo '         (default: python)'
  echo '  --mpirun-cmd CMD or --mpirun-cmd=CMD: specify mpirun command'
  echo '         (default: mpirun)'
  echo '  --device DEVICE or --device=DEVICE: specify test device'
  echo '         available DEVICE are [ve_vh|ve|vh]'
  echo '         (default: ve_vh)'
  echo '  --hosts HOST1,HOST2,･･･ or --hosts=HOST1,HOST2,･･･: specify VH hosts'
  echo '         (default: None)'
}

while (( $# > 0 ))
do
    case $1 in
        --test | --test=*)
            if [[ "$1" =~ ^--test= ]]; then
                TEST=$(echo $1 | sed -e 's/^--test=//')
            elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
                usage
                exit 1
            else
                TEST="$2"
                shift
            fi
            if [[ $TEST != "full" ]] && [[ $TEST != "small" ]]; then
                usage
                exit 1
            fi
        ;;
        --np | --np=*)
            if [[ "$1" =~ ^--np= ]]; then
                NP=$(echo $1 | sed -e 's/^--np=//')
            elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
                usage
                exit 1
            else
                NP="$2"
                shift
            fi
            if [[ ${NP} =~ ^[0-9]+$ ]]; then
                NP=${NP}
            else
                usage
                exit 1
            fi
        ;;
        --python-cmd | --python-cmd=*)
            if [[ "$1" =~ ^--python-cmd= ]]; then
                PYTHON_CMD=$(echo $1 | sed -e 's/^--python-cmd=//')
            elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
                usage
                exit 1
            else
                PYTHON_CMD="$2"
                shift
            fi
            if [[ -z ${PYTHON_CMD} ]]; then
                usage
                exit 1
            fi
        ;;
        --mpirun-cmd | --mpirun-cmd=*)
            if [[ "$1" =~ ^--mpirun-cmd= ]]; then
                MPIRUN_CMD=$(echo $1 | sed -e 's/^--mpirun-cmd=//')
            elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
                usage
                exit 1
            else
                MPIRUN_CMD="$2"
                shift
            fi
            if [[ -z ${MPIRUN_CMD} ]]; then
                usage
                exit 1
            fi
        ;;
        --device | --device=*)
            if [[ "$1" =~ ^--device= ]]; then
                TEST_DEVICE=$(echo $1 | sed -e 's/^--device=//')
            elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
                usage
                exit 1
            else
                TEST_DEVICE="$2"
                shift
            fi
            if [[ $TEST_DEVICE != "ve_vh" ]] && [[ $TEST_DEVICE != "ve" ]] && [[ $TEST_DEVICE != "vh" ]]; then
                usage
                exit 1
            fi
        ;;
        --hosts | --hosts=*)
            if [[ "$1" =~ ^--hosts= ]]; then
                HOSTS=$(echo $1 | sed -e 's/^--hosts=//')
            elif [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
                usage
                exit 1
            else
                HOSTS="$2"
                shift
            fi
            if [[ -z ${HOSTS} ]]; then
                usage
                exit 1
            fi
            HOSTS_OPT='-hosts '${HOSTS}
        ;;
        -h | --help)
            usage
            exit 1
        ;;
    esac
    shift
done

echo 'VE_NLCPY_NODELIST='${VE_NLCPY_NODELIST}

export NMPI_USE_COMMAND_SEARCH_PATH=ON
set -x
MPI4PYVE_TEST_PATTERN=${TEST} MPI4PYVE_TEST_DEVICE=${TEST_DEVICE} ${MPIRUN_CMD} ${HOSTS_OPT} -veo -np ${NP} ${PYTHON_CMD} test_coverage_device_comm.py
MPI4PYVE_TEST_PATTERN=${TEST} MPI4PYVE_TEST_DEVICE=${TEST_DEVICE} ${MPIRUN_CMD} ${HOSTS_OPT} -veo -np ${NP} ${PYTHON_CMD} test_coverage_device_file.py
MPI4PYVE_TEST_PATTERN=${TEST} MPI4PYVE_TEST_DEVICE=${TEST_DEVICE} ${MPIRUN_CMD} ${HOSTS_OPT} -veo -np ${NP} ${PYTHON_CMD} test_coverage_device_win.py
MPI4PYVE_TEST_PATTERN=${TEST} MPI4PYVE_TEST_DEVICE=${TEST_DEVICE} ${MPIRUN_CMD} ${HOSTS_OPT} -veo -np ${NP} ${PYTHON_CMD} test_coverage_device_datatype.py
MPI4PYVE_TEST_PATTERN=${TEST} MPI4PYVE_TEST_DEVICE=${TEST_DEVICE} ${MPIRUN_CMD} ${HOSTS_OPT} -veo -np ${NP} ${PYTHON_CMD} test_coverage_device_message.py
set +x

