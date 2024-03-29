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

"""Implements MPIPoolExecutor."""

import time
import functools
import itertools
import threading

from . import Future
from . import Executor
from . import as_completed

from . import _lib

def raise_notimpl_for_necmpi(func):
    def _raise_wrapper(*args, **kwargs):
        raise NotImplementedError('%s on mpi4py-ve is not implemented yet.'
                                  %func)
    return _raise_wrapper

class MPIPoolExecutor(Executor):
    """MPI-based asynchronous executor."""

    Future = Future

    @raise_notimpl_for_necmpi('MPIPoolExecutor')
    def __init__(self, max_workers=None, **kwargs):
        """Initialize a new MPIPoolExecutor instance.

        Args:
            max_workers: The maximum number of MPI processes that can be used
                to execute the given calls. If ``None`` or not given then the
                number of worker processes will be determined from the MPI
                universe size attribute if defined, otherwise a single worker
                process will be spawned.

        Keyword Args:
            python_exe: Path to Python executable used to spawn workers.
            python_args: Command line arguments to pass to Python executable.
            mpi_info: Dict or iterable with ``(key, value)`` pairs.
            globals: Dict or iterable with global variables to set in workers.
            main: If ``False``, do not import ``__main__`` in workers.
            path: List of paths to append to ``sys.path`` in workers.
            wdir: Path to set current working directory in workers.
            env: Environment variables to update ``os.environ`` in workers.

        """
        if max_workers is not None:
            max_workers = int(max_workers)
            if max_workers <= 0:
                raise ValueError("max_workers must be greater than 0")
            kwargs['max_workers'] = max_workers

        self._options = kwargs
        self._shutdown = False
        self._lock = threading.Lock()
        self._pool = None

    _make_pool = staticmethod(_lib.WorkerPool)

    def _bootstrap(self):
        if self._pool is None:
            self._pool = self._make_pool(self)

    def bootup(self, wait=True):
        """Allocate executor resources eagerly.

        Args:
            wait: If ``True`` then bootup will not return until the
                executor resources are ready to process submissions.

        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot bootup after shutdown")
            self._bootstrap()
            if wait:
                self._pool.wait()
            return self

    def submit(self, fn, *args, **kwargs):
        """Submit a callable to be executed with the given arguments.

        Schedule the callable to be executed as ``fn(*args, **kwargs)`` and
        return a `Future` instance representing the execution of the callable.

        Returns:
            A `Future` representing the given call.

        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot submit after shutdown")
            self._bootstrap()
            future = self.Future()
            task = (fn, args, kwargs)
            self._pool.push((future, task))
            return future

    def map(self, fn, *iterables, **kwargs):
        """Return an iterator equivalent to ``map(fn, *iterables)``.

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            iterables: Iterables yielding positional arguments to be passed to
                the callable.
            timeout: The maximum number of seconds to wait. If ``None``, then
                there is no limit on the wait time.
            chunksize: The size of the chunks the iterable will be broken into
                before being passed to a worker process.

        Keyword Args:
            unordered: If ``True``, yield results out-of-order, as completed.

        Returns:
            An iterator equivalent to built-in ``map(func, *iterables)``
            but the calls may be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If ``fn(*args)`` raises for any values.

        """
        # pylint: disable=arguments-differ
        iterable = getattr(itertools, 'izip', zip)(*iterables)
        return self.starmap(fn, iterable, **kwargs)

    def starmap(self, fn, iterable, timeout=None, chunksize=1, **kwargs):
        """Return an iterator equivalent to ``itertools.starmap(...)``.

        Args:
            fn: A callable that will take positional argument from iterable.
            iterable: An iterable yielding ``args`` tuples to be used as
                positional arguments to call ``fn(*args)``.
            timeout: The maximum number of seconds to wait. If ``None``, then
                there is no limit on the wait time.
            chunksize: The size of the chunks the iterable will be broken into
                before being passed to a worker process.

        Keyword Args:
            unordered: If ``True``, yield results out-of-order, as completed.

        Returns:
            An iterator equivalent to ``itertools.starmap(fn, iterable)``
            but the calls may be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If ``fn(*args)`` raises for any values.

        """
        unordered = kwargs.pop('unordered', False)
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")
        if chunksize == 1:
            return _starmap_helper(self.submit, fn, iterable,
                                   timeout, unordered)
        else:
            return _starmap_chunks(self.submit, fn, iterable,
                                   timeout, unordered, chunksize)

    def shutdown(self, wait=True):
        """Clean-up the resources associated with the executor.

        It is safe to call this method several times. Otherwise, no other
        methods can be called after this one.

        Args:
            wait: If ``True`` then shutdown will not return until all running
                futures have finished executing and the resources used by the
                executor have been reclaimed.

        """
        with self._lock:
            if not self._shutdown:
                self._shutdown = True
                if self._pool is not None:
                    self._pool.done()
            if wait:
                if self._pool is not None:
                    self._pool.join()
                    self._pool = None


def _starmap_helper(submit, function, iterable, timeout, unordered):
    if timeout is not None:
        timer = time.time
        end_time = timeout + timer()

    futures = [submit(function, *args) for args in iterable]
    if unordered:
        futures = set(futures)

    def result_iterator():  # pylint: disable=missing-docstring
        try:
            if unordered:
                if timeout is None:
                    iterator = as_completed(futures)
                else:
                    iterator = as_completed(futures, end_time - timer())
                for future in iterator:
                    futures.remove(future)
                    future = [future]
                    yield future.pop().result()
            else:
                futures.reverse()
                if timeout is None:
                    while futures:
                        yield futures.pop().result()
                else:
                    while futures:
                        yield futures.pop().result(end_time - timer())
        except:
            while futures:
                futures.pop().cancel()
            raise
    return result_iterator()


def _apply_chunks(function, chunk):
    return [function(*args) for args in chunk]


def _build_chunks(chunksize, iterable):
    iterable = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(iterable, chunksize))
        if not chunk:
            return
        yield (chunk,)


def _chain_from_iterable_of_lists(iterable):
    for item in iterable:
        item.reverse()
        while item:
            yield item.pop()


def _starmap_chunks(submit, function, iterable,
                    timeout, unordered, chunksize):
    # pylint: disable=too-many-arguments
    function = functools.partial(_apply_chunks, function)
    iterable = _build_chunks(chunksize, iterable)
    result = _starmap_helper(submit, function, iterable,
                             timeout, unordered)
    return _chain_from_iterable_of_lists(result)


class MPICommExecutor(object):
    """Context manager for `MPIPoolExecutor`.

    This context manager splits a MPI (intra)communicator in two
    disjoint sets: a single master process and the remaining worker
    processes. These sets are then connected through an intercommunicator.
    The target of the ``with`` statement is assigned either an
    `MPIPoolExecutor` instance (at the master) or ``None`` (at the workers).

    Example::

        with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
            if executor is not None: # master process
                executor.submit(...)
                executor.map(...)
    """

    # pylint: disable=too-few-public-methods

    @raise_notimpl_for_necmpi('MPICommExecutor')
    def __init__(self, comm=None, root=0, **kwargs):
        """Initialize a new MPICommExecutor instance.

        Args:
            comm: MPI (intra)communicator.
            root: Designated master process.

        Raises:
            ValueError: If the communicator has wrong kind or
               the root value is not in the expected range.

        """
        if comm is None:
            comm = _lib.get_comm_world()
        if comm.Is_inter():
            raise ValueError("Expecting an intracommunicator")
        if root < 0 or root >= comm.Get_size():
            raise ValueError("Expecting root in range(comm.size)")

        self._comm = comm
        self._root = root
        self._options = kwargs
        self._executor = None

    def __enter__(self):
        """Return `MPIPoolExecutor` instance at the root."""
        # pylint: disable=protected-access
        if self._executor is not None:
            raise RuntimeError("__enter__")

        comm = self._comm
        root = self._root
        options = self._options
        executor = None

        if _lib.SharedPool:
            assert root == 0
            executor = MPIPoolExecutor(**options)
            executor._pool = _lib.SharedPool(executor)
        elif comm.Get_size() == 1:
            executor = MPIPoolExecutor(**options)
            executor._pool = _lib.ThreadPool(executor)
        elif comm.Get_rank() == root:
            executor = MPIPoolExecutor(**options)
            executor._pool = _lib.SplitPool(executor, comm, root)
        else:
            _lib.server_main_split(comm, root, **options)

        self._executor = executor
        return executor

    def __exit__(self, *args):
        """Shutdown `MPIPoolExecutor` instance at the root."""
        executor = self._executor
        self._executor = None

        if executor is not None:
            executor.shutdown(wait=True)
            return False
        else:
            return True


class ThreadPoolExecutor(MPIPoolExecutor):  # noqa: D204
    """`MPIPoolExecutor` subclass using a pool of threads."""
    _make_pool = staticmethod(_lib.ThreadPool)


class ProcessPoolExecutor(MPIPoolExecutor):  # noqa: D204
    """`MPIPoolExecutor` subclass using a pool of processes."""
    _make_pool = staticmethod(_lib.SpawnPool)
