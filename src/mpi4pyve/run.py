### mpi4py-ve License ##
#
#  Copyright (c) 2022, NEC Corporation.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification,
#  are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice, this
#     list of conditions and the following disclaimer listed in this license in the
#     documentation and/or other materials provided with the distribution.
#
# The copyright holders provide no reassurances that the source code provided does not
# infringe any patent, copyright, or any other intellectual property rights of third
# parties. The copyright holders disclaim any liability to any recipient for claims
# brought against recipient by any third party for infringement of that parties
# intellectual property rights.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANYTHEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# NOTE: This code is derived from mpi4py written by Lisandro Dalcin.
#
### mpi4py License ##
#
#  Copyright (c) 2019, Lisandro Dalcin. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification, 
#  are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Run Python code using ``mpi4pyve``.

Run Python code (scripts, modules, zip files) using the ``runpy``
module. In case of an unhandled exception, abort execution of the MPI
program by calling ``MPI.COMM_WORLD.Abort()``.
"""
from __future__ import print_function


def run_command_line(args=None):
    """Run command line ``[pyfile | -m mod | -c cmd | -] [arg] ...``.

    * ``pyfile`` : program read from script file
    * ``-m mod`` : run library module as a script
    * ``-c cmd`` : program passed in as a command string
    * ``-``      : program read from standard input (``sys.stdin``)
    * ``arg ...``: arguments passed to program in ``sys.argv[1:]``
    """
    # pylint: disable=missing-docstring
    import sys
    from runpy import run_module, run_path

    def run_string(string, init_globals=None, run_name=None,
                   filename='<string>', argv0='-c'):
        from runpy import _run_module_code
        karg = 'script_name' if sys.version_info >= (3, 4) else 'mod_fname'
        code = compile(string, filename, 'exec', 0, 1)
        return _run_module_code(code, init_globals, run_name, **{karg: argv0})

    sys.argv[:] = args if args is not None else sys.argv[1:]

    if sys.argv[0] == '-':
        cmd = sys.stdin.read()
        run_string(cmd, run_name='__main__', filename='<stdin>', argv0='-')
    elif sys.argv[0] == '-c':
        cmd = sys.argv.pop(1)  # Remove "cmd" from argument list
        run_string(cmd, run_name='__main__', filename='<string>', argv0='-c')
    elif sys.argv[0] == '-m':
        del sys.argv[0]  # Remove "-m" from argument list
        run_module(sys.argv[0], run_name='__main__', alter_sys=True)
    else:
        from os.path import realpath, dirname
        if not getattr(sys.flags, 'isolated', 0):  # pragma: no branch
            sys.path[0] = realpath(dirname(sys.argv[0]))  # Fix sys.path
        run_path(sys.argv[0], run_name='__main__')


def set_abort_status(status):
    """Terminate MPI execution environment at Python exit.

    Terminate MPI execution environment at Python exit by calling
    ``MPI.COMM_WORLD.Abort(status)``. This function should be called
    within an ``except`` block. Afterwards, exceptions should be
    re-raised.
    """
    import sys
    status = (status if isinstance(status, int)
              else 0 if status is None else 1)
    pkg = __package__ or __name__.rpartition('.')[0]
    mpi = sys.modules.get(pkg + '.MPI')
    if mpi is not None and status:
        # pylint: disable=protected-access
        mpi._set_abort_status(status)
    return sys.exc_info()


def main():
    """Entry-point for ``python -m mpi4pyve.run ...``."""
    # pylint: disable=missing-docstring
    import os
    import sys

    def version():
        from . import __version__
        print(__package__, __version__, file=sys.stdout)
        sys.exit(0)

    def usage(errmess=None):
        from textwrap import dedent
        if __name__ == '__main__':
            prog_name = __package__ + '.run'
        else:
            prog_name = __package__
        python_exe = os.path.basename(sys.executable)
        subs = dict(prog=prog_name, python=python_exe)

        cmdline = dedent("""
        usage: {python} -m {prog} [options] <pyfile> [arg] ...
           or: {python} -m {prog} [options] -m <mod> [arg] ...
           or: {python} -m {prog} [options] -c <cmd> [arg] ...
           or: {python} -m {prog} [options] - [arg] ...
        """).strip().format(**subs)

        helptip = dedent("""
        Try `{python} -m {prog} -h` for more information.
        """).strip().format(**subs)

        options = dedent("""
        options:
          --version            show version number and exit
          -h|--help            show this help message and exit
          -rc <key=value,...>  set 'mpi4pyve.rc.key=value'
          -p|--profile <pmpi>  use <pmpi> for profiling
          --mpe                profile with MPE
          --vt                 profile with VampirTrace
        """).strip()

        if errmess:
            print(errmess, file=sys.stderr)
            print(cmdline, file=sys.stderr)
            print(helptip, file=sys.stderr)
            sys.exit(1)
        else:
            print(cmdline, file=sys.stdout)
            print(options, file=sys.stdout)
            sys.exit(0)

    def parse_command_line(args=None):
        # pylint: disable=too-many-branches

        class Options(object):
            # pylint: disable=too-few-public-methods
            rc_args = {}
            profile = None

        def poparg(args):
            if len(args) < 2 or args[1].startswith('-'):
                usage('Argument expected for option: ' + args[0])
            return args.pop(1)

        options = Options()
        args = sys.argv[1:] if args is None else args[:]
        while args and args[0].startswith('-'):
            if args[0] in ('-m', '-c', '-'):
                break  # Stop processing options
            if args[0] in ('-h', '-help', '--help'):
                usage()  # Print help and exit
            if args[0] in ('-version', '--version'):
                version()  # Print version and exit
            try:
                arg0 = args[0]
                if arg0.startswith('--'):
                    if '=' in arg0:
                        i = arg0.index('=')
                        opt, arg = arg0[1:i], arg0[i+1:]
                        if opt in ('-rc', '-profile'):
                            arg0, args[1:1] = opt, [arg]
                    else:
                        arg0 = arg0[1:]
                if arg0 == '-rc':
                    for entry in poparg(args).split(','):
                        i = entry.index('=')
                        key = entry[:i].strip()
                        val = entry[i+1:].strip()
                        if not key or not val:
                            raise ValueError(entry)
                        try:
                            # pylint: disable=eval-used
                            options.rc_args[key] = eval(val, {})
                        except NameError:
                            options.rc_args[key] = val
                elif arg0 in ('-p', '-profile'):
                    options.profile = poparg(args) or None
                elif arg0 in ('-mpe', '-vt'):
                    options.profile = arg0[1:]
                else:
                    usage('Unknown option: ' + args[0])
                del args[0]
            except Exception:  # pylint: disable=broad-except
                # Bad option, print usage and exit with error
                usage('Cannot parse option: ' + args[0])
        # Check remaining args and return to caller
        if len(args) < 1:
            usage("No path specified for execution")
        elif args[0] in ('-m', '-c') and len(args) < 2:
            usage("Argument expected for option: " + args[0])
        return options, args

    def bootstrap(options):
        if options.rc_args:  # Set mpi4pyve.rc parameters
            from . import rc
            rc(**options.rc_args)
        if options.profile:  # Load profiling library
            from . import profile
            profile(options.profile)

    # Parse and process command line options
    options, args = parse_command_line()
    bootstrap(options)

    # Run user code. In case of an unhandled exception, abort
    # execution of the MPI program by calling 'MPI_Abort()'.
    try:
        run_command_line(args)
    except SystemExit as exc:
        set_abort_status(exc.code)
        raise
    except:
        set_abort_status(1)
        raise


if __name__ == '__main__':
    main()
