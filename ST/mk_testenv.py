import os
import sys
import shutil
import string
import subprocess
import datetime
from params import url, branch, use_wrapper, nlc_ver, mpi_ver, ques, \
    tests, qsub, python

# specify test directory and run/submit script for pytest
test_dir = 'coverage_device_test'
go_script = 'go.sh'

_qsub_template = r'''
#!/bin/sh
#PBS -N ${job_name}
#PBS -o ${stdout_file}
#PBS -e ${stderr_file}
#PBS -l elapstim_req=${elapstim_req}
#PBS -q ${que_name}
#PBS -T necmpi
#PBS -b 2               # The number of logical hosts
#PBS --venum-lhost=4    # The number of VEs per logical host
#PBS --cpunum-lhost=${cpunum_job}   # The number of CPUs per logical host
#PBS --use-hca=2
${pbs_add_opt}

cd $${PBS_O_WORKDIR:-$${QSUB_WORKDIR:-.}}

${nlc_source}
${mpi_source}
${mpi4pyvevars_source}

export NMPI_USE_COMMAND_SEARCH_PATH=ON

echo "NLCPy version:" `${python} -c "import nlcpy; print(nlcpy.__version__)"`
echo "NLCPy path   :" `${python} -c "import nlcpy; print(nlcpy.__path__)"`
echo "mpi4py-ve version  :" `${mpirun} -veo -np 1 ${python} \
                         -c "import mpi4pyve; print(mpi4pyve.__version__)"`
echo "mpi4py-ve path     :" `${mpirun} -veo -np 1 ${python} \
                         -c "import mpi4pyve; print(mpi4pyve.__path__)"`
echo "mpi4py-ve proc name:" `${mpirun} -veo -np 8 ${python} \
                         -c "from mpi4pyve import MPI; print(MPI.Get_processor_name())"`

echo "Start date   : " `LANG=C date`
(cd coverage_device_test; sh run.sh --test=${test} --sub-test=${sub_test} \
                         --np=${np} --python-cmd=${python})
echo "End date     : " `LANG=C date`
'''


def _exec_cmd(cmd, log_stream=None):
    try:
        env = os.environ
        log = subprocess.check_output(cmd, env=env,
                                      stderr=subprocess.STDOUT,
                                      universal_newlines=True)
        if log_stream is not None:
            msg = 'cmd: {}\n'.format(' '.join(cmd)) + log
            log_stream.write(msg)
        return log
    except subprocess.CalledProcessError as e:
        msg = ('`{0}` command returns non-zero exit status. \n'
               'command: {1}\n'
               'return-code: {2}\n'
               'stdout/stderr: \n'
               '{3}'.format(' '.join(cmd),
                            e.cmd,
                            e.returncode,
                            e.output))
        if log_stream is not None:
            log_stream.write(msg)
        raise
    except Exception:
        raise


if __name__ == '__main__':
    if python == '':
        python = '/usr/bin/python3'
    if qsub == '':
        qsub = '/opt/nec/nqsv/bin/qsub'

    # initialize default values
    nlc_ver_default = nlc_ver
    mpi_ver_default = mpi_ver
    branch_default = branch
    use_wrapper_default = use_wrapper

    for q in ques:
        # set default values
        branch = branch_default
        nlc_ver = nlc_ver_default
        mpi_ver = mpi_ver_default
        use_wrapper = use_wrapper_default

        # override default options with queue local values.
        if q['branch'] != '':
            branch = q['branch']
        if q['nlc_ver'] != '':
            nlc_ver = q['nlc_ver']
        if q['mpi_ver'] != '':
            mpi_ver = q['mpi_ver']
        if q['use_wrapper'] != '':
            use_wrapper = q['use_wrapper']

        # setup nlcvars/necmpivars scripts path
        nlc_source_ve1 = 'source /opt/nec/ve/nlc/{}/bin/nlcvars.sh'.format(nlc_ver)
        nlc_source_ve3 = 'source /opt/nec/ve3/nlc/{}/bin/nlcvars.sh'.format(nlc_ver)
        mpi_source_ve1 = 'source /opt/nec/ve/mpi/{}/bin/necmpivars.sh gnu 4.8.5'. \
                         format(mpi_ver)
        mpi_source_ve3 = 'source /opt/nec/ve3/mpi/{}/bin/necmpivars.sh gnu 4.8.5'. \
                         format(mpi_ver)
        if int(use_wrapper) != 0:
            # mpi4py-ve version 1.1.0 or later requires mpirun wrapper
            mpi4pyvevars_source = 'source `$(which python) -c \"import mpi4pyve;' \
                                  'print(mpi4pyve.__path__[0])\"`/bin/mpi4pyvevars.sh'
            mpirun = 'mpirun-py'
        else:
            mpi4pyvevars_source = ''
            mpirun = 'mpirun'

        # clone specified branch if not yet cloned.
        repos_dir = 'mpi4pyverepos.' + branch
        # replace directory separator characters "/" -> "_"
        repos_dir = repos_dir.replace('/', '_')
        if os.path.isdir(repos_dir):
            print(repos_dir + ' found, reusing it.')
        else:
            print(repos_dir + ' not found, cloning.')
            # clone only 'tests' directory
            # old server does not recognize --filter option, just ignore it.
            cmd = ['git', 'clone', '--filter=blob:none', '--sparse', url,
                   '-b', branch, repos_dir]
            _exec_cmd(cmd, log_stream=sys.stdout)
            os.chdir(repos_dir)
            cmd = ['git', 'sparse-checkout', 'set', test_dir]
            _exec_cmd(cmd, log_stream=sys.stdout)
            os.chdir('..')

        # copy mpi4pyverepos.(branch)/coverage_device_test directory
        if q['dir_name'] == '':
            dir_name = q['name']
        else:
            dir_name = q['dir_name']
        print('processing ' + q['name'] + ' for directory ' + dir_name + '...')
        os.mkdir(dir_name)
        shutil.copytree('./' + repos_dir + '/' + test_dir, dir_name +
                        '/' + test_dir, symlinks=True)
        if q['ve_arch'] == 1:
            nlc_source = nlc_source_ve1
            mpi_source = mpi_source_ve1
        elif q['ve_arch'] == 3:
            nlc_source = nlc_source_ve3
            mpi_source = mpi_source_ve3
        else:
            raise ValueError('Unknown ve_arch:', q['ve_arch'])
        if q['add_opt'] != '':
            pbs_add_opt = '#PBS ' + q['add_opt']
        else:
            pbs_add_opt = ''

        datestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # write qsub.*.sh
        for t in tests:
            qsub_script = string.Template(_qsub_template).substitute(
                job_name='"SLG mpi4py-ve ' + t['sub_test'] + '"',
                stdout_file='qsub.mpi4py-ve.' + t['sub_test'] + '.' + datestr + '.out',
                stderr_file='qsub.mpi4py-ve.' + t['sub_test'] + '.' + datestr + '.err',
                elapstim_req=t['elapstim_req'],
                que_name=q['name'],
                cpunum_job=q['cpunum_job'],
                pbs_add_opt=pbs_add_opt,
                nlc_source=nlc_source,
                mpi_source=mpi_source,
                mpi4pyvevars_source=mpi4pyvevars_source,
                mpirun=mpirun,
                test=t['test'],
                sub_test=t['sub_test'],
                np=t['np'],
                python=python
            )
            with open(dir_name + '/qsub.' + t['sub_test'] + '.sh', 'w') as f:
                f.write(qsub_script)

        # write go.sh
        with open(dir_name + '/' + go_script, 'w') as f:
            f.write('#!/bin/sh\n')
            for t in tests:
                f.write(qsub + ' ./qsub.' + t['sub_test'] + '.sh\n')
