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
"""Run MPI benchmarks and tests."""
import sys as _sys


def helloworld(comm, args=None, verbose=True):
    """Hello, World! using MPI."""
    from argparse import ArgumentParser
    parser = ArgumentParser(prog=__package__ + ".bench helloworld")
    parser.add_argument("-q", "--quiet", action="store_false",
                        dest="verbose", default=verbose)
    options = parser.parse_args(args)

    from . import MPI
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    message = ("Hello, World! I am process %*d of %d on %s.\n"
               % (len(str(size - 1)), rank, size, name))
    comm.Barrier()
    if rank > 0:
        comm.Recv([None, 'B'], rank - 1)
    if options.verbose:
        _sys.stdout.write(message)
        _sys.stdout.flush()
    if rank < size - 1:
        comm.Send([None, 'B'], rank + 1)
    comm.Barrier()
    return message


def ringtest(comm, args=None, verbose=True):
    """Time a message going around the ring of processes."""
    # pylint: disable=too-many-locals
    from argparse import ArgumentParser
    parser = ArgumentParser(prog=__package__ + ".bench ringtest")
    parser.add_argument("-q", "--quiet", action="store_false",
                        dest="verbose", default=verbose)
    parser.add_argument("-n", "--size", type=int, default=1, dest="size",
                        help="message size")
    parser.add_argument("-s", "--skip", type=int, default=0, dest="skip",
                        help="number of warm-up iterations")
    parser.add_argument("-l", "--loop", type=int, default=1, dest="loop",
                        help="number of iterations")
    options = parser.parse_args(args)

    def ring(comm, n=1, loop=1, skip=0):
        # pylint: disable=invalid-name
        # pylint: disable=missing-docstring
        from array import array
        from . import MPI
        iterations = list(range((loop + skip)))
        size = comm.Get_size()
        rank = comm.Get_rank()
        source = (rank - 1) % size
        dest = (rank + 1) % size
        Sendrecv = comm.Sendrecv
        Send = comm.Send
        Recv = comm.Recv
        Wtime = MPI.Wtime
        sendmsg = array('B', [+42]) * n
        recvmsg = array('B', [0x0]) * n
        if size == 1:
            for i in iterations:
                if i == skip:
                    tic = Wtime()
                Sendrecv(sendmsg, dest, 0,
                         recvmsg, source, 0)
        else:
            if rank == 0:
                for i in iterations:
                    if i == skip:
                        tic = Wtime()
                    Send(sendmsg, dest, 0)
                    Recv(recvmsg, source, 0)
            else:
                sendmsg = recvmsg
                for i in iterations:
                    if i == skip:
                        tic = Wtime()
                    Recv(recvmsg, source, 0)
                    Send(sendmsg, dest, 0)
        toc = Wtime()
        if comm.rank == 0 and sendmsg != recvmsg:  # pragma: no cover
            import warnings
            import traceback
            try:
                warnings.warn("received message does not match!")
            except UserWarning:
                traceback.print_exc()
                comm.Abort(2)
        return toc - tic

    size = getattr(options, 'size', 1)
    loop = getattr(options, 'loop', 1)
    skip = getattr(options, 'skip', 0)
    comm.Barrier()
    elapsed = ring(comm, size, loop, skip)
    if options.verbose and comm.rank == 0:
        message = ("time for %d loops = %g seconds (%d processes, %d bytes)\n"
                   % (loop, elapsed, comm.size, size))
        _sys.stdout.write(message)
        _sys.stdout.flush()
    return elapsed


def main(args=None):
    """Entry-point for ``python -m mpi4pyve.bench``."""
    from argparse import ArgumentParser, REMAINDER
    parser = ArgumentParser(prog=__package__ + ".bench",
                            usage="%(prog)s [options] <command> [args]")
    parser.add_argument("--threads",
                        action="store_true", dest="threads", default=None,
                        help="initialize MPI with thread support")
    parser.add_argument("--no-threads",
                        action="store_false", dest="threads", default=None,
                        help="initialize MPI without thread support")
    parser.add_argument("--thread-level",
                        dest="thread_level", default=None,
                        action="store", metavar="LEVEL",
                        choices="single funneled serialized multiple".split(),
                        help="initialize MPI with required thread level")
    parser.add_argument("--mpe",
                        action="store_true", dest="mpe", default=False,
                        help="use MPE for MPI profiling")
    parser.add_argument("--vt",
                        action="store_true", dest="vt", default=False,
                        help="use VampirTrace for MPI profiling")
    parser.add_argument("command",
                        action="store", metavar="<command>",
                        help="benchmark command to run")
    parser.add_argument("args",
                        nargs=REMAINDER, metavar="[args]",
                        help="arguments for benchmark command")
    options = parser.parse_args(args)

    from . import rc, profile
    if options.threads is not None:
        rc.threads = options.threads
    if options.thread_level is not None:
        rc.thread_level = options.thread_level
    if options.mpe:
        profile('mpe', logfile='mpi4pyve')
    if options.vt:
        profile('vt', logfile='mpi4pyve')

    from . import MPI
    comm = MPI.COMM_WORLD
    if options.command not in main.commands:
        if comm.rank == 0:
            parser.error("unknown command '%s'" % options.command)
        parser.exit(2)
    command = main.commands[options.command]
    command(comm, options.args)
    parser.exit()

main.commands = {
    'helloworld': helloworld,
    'ringtest': ringtest,
}

if __name__ == '__main__':
    main()
