url = 'https://sfdjp11702.swf.nec.co.jp/gitlab/slg/mpi4pyverepos.git'
# branch = 'rhel10/numpy1.26.0-pyproject'
# branch = 'develop'
branch = 'release'

nlc_ver = '3.1.0'
mpi_ver = '3.11.0'

use_wrapper = 0

qsub = '/opt/nec/nqsv/bin/qsub'
python = '/usr/bin/python3'

ques = [
    # NLC 3.1.0
    {'name': 'numa',  'dir_name': '',
                      've_arch': 1, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'smp',   'dir_name': '',
                      've_arch': 1, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'numa2', 'dir_name': '',
                      've_arch': 1, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'smp2',  'dir_name': '',
                      've_arch': 1, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'numa3', 'dir_name': '',
                      've_arch': 3, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'smp3',  'dir_name': '',
                      've_arch': 3, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'turin', 'dir_name': '',
                      've_arch': 3, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : 1,
                      'nlc_ver' : '', 'mpi_ver' : '', 'branch' : 'rhel10/numpy1.26.0-pyproject'},

    # NLC 3.1.0, VE1 on VE3
    {'name': 'numa3', 'dir_name': 'numa3.ve1',
                      've_arch': 1, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'smp3',  'dir_name': 'smp3.ve1',
                      've_arch': 1, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'turin', 'dir_name': 'turin.ve1',
                      've_arch': 1, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : 1,
                      'nlc_ver' : '', 'mpi_ver' : '', 'branch' : 'rhel10/numpy1.26.0-pyproject'},

    # NLC 3.0.0
    {'name': 'numa',  'dir_name': 'numa.nlc300',
                      've_arch': 1, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '3.0.0', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'smp',   'dir_name': 'smp.nlc300',
                      've_arch': 1, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '3.0.0', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'numa2', 'dir_name': 'numa2.nlc300',
                      've_arch': 1, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '3.0.0', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'smp2',  'dir_name': 'smp2.nlc300',
                      've_arch': 1, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '3.0.0', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'numa3', 'dir_name': 'numa3.nlc300',
                      've_arch': 3, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '3.0.0', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'smp3',  'dir_name': 'smp3.nlc300',
                      've_arch': 3, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '3.0.0', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'turin', 'dir_name': 'turin.nlc300',
                      've_arch': 3, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : 1,
                      'nlc_ver' : '3.0.0', 'mpi_ver' : '', 'branch' : 'rhel10/numpy1.26.0-pyproject'},

    # NLC 3.0.0, VE1 on VE3
    {'name': 'numa3', 'dir_name': 'numa3.nlc300.ve1',
                      've_arch': 1, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '3.0.0', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'smp3',  'dir_name': 'smp3.nlc300.ve1',
                      've_arch': 1, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : '',
                      'nlc_ver' : '3.0.0', 'mpi_ver' : '', 'branch' : ''},
    {'name': 'turin', 'dir_name': 'turin.nlc300.ve1',
                      've_arch': 1, 'cpunum_job': 4, 'add_opt': '', 'use_wrapper' : 1,
                      'nlc_ver' : '3.0.0', 'mpi_ver' : '', 'branch' : 'rhel10/numpy1.26.0-pyproject'},
]

# parameters for run.sh and elapsetim_req
tests = [
    {'test': 'small', 'sub_test': 'comm', 'np': 8, 'elapstim_req': 3000},
    {'test': 'small', 'sub_test': 'file', 'np': 8, 'elapstim_req': 3400},
    {'test': 'small', 'sub_test': 'win', 'np': 8, 'elapstim_req': 3000},
    {'test': 'small', 'sub_test': 'datatype', 'np': 8, 'elapstim_req': 3000},
    {'test': 'small', 'sub_test': 'message', 'np': 8, 'elapstim_req': 3000},
]

