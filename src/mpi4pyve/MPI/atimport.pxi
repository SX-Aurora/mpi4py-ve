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

# -----------------------------------------------------------------------------

cdef extern from "atimport.h": pass

# -----------------------------------------------------------------------------

cdef extern from "Python.h":
    enum: PY3 "(PY_MAJOR_VERSION>=3)"
    enum: PY2 "(PY_MAJOR_VERSION==2)"
    enum: PYPY "PyMPI_RUNTIME_PYPY"

    void PySys_WriteStderr(char*,...)
    int Py_AtExit(void (*)())

    ctypedef struct PyObject
    PyObject *Py_None
    void Py_CLEAR(PyObject*)

    void Py_INCREF(object)
    void Py_DECREF(object)

# -----------------------------------------------------------------------------

cdef extern from *:
    enum: USE_MATCHED_RECV "PyMPI_USE_MATCHED_RECV"

ctypedef struct Options:
    int initialize
    int threads
    int thread_level
    int finalize
    int fast_reduce
    int recv_mprobe
    int errors

cdef Options options
options.initialize = 1
options.threads = 1
options.thread_level = MPI_THREAD_SERIALIZED
options.finalize = 1
options.fast_reduce = 1
options.recv_mprobe = 1
options.errors = 1

cdef int warnOpt(object name, object value) except -1:
    cdef object warn
    from warnings import warn
    warn("mpi4pyve.rc: '%s': unexpected value '%r'" % (name, value))

cdef int getOptions(Options* opts) except -1:
    cdef object rc
    opts.initialize = 1
    opts.threads = 1
    opts.thread_level = MPI_THREAD_SERIALIZED
    opts.finalize = 1
    opts.fast_reduce = 1
    opts.recv_mprobe = 1
    opts.errors = 1
    try: from mpi4pyve import rc
    except: return 0
    #
    cdef object initialize = True
    cdef object threads = True
    cdef object thread_level = 'serialized'
    cdef object finalize = None
    cdef object fast_reduce = True
    cdef object recv_mprobe = True
    cdef object errors = 'exception'
    try: initialize = rc.initialize
    except: pass
    try: threads = rc.threads
    except: pass
    try: threads = rc.threaded # backward
    except: pass               # compatibility
    try: thread_level = rc.thread_level
    except: pass
    try: finalize = rc.finalize
    except: pass
    try: fast_reduce = rc.fast_reduce
    except: pass
    try: recv_mprobe = rc.recv_mprobe
    except: pass
    try: errors = rc.errors
    except: pass
    #
    if initialize in (True, 'yes'):
        opts.initialize = 1
    elif initialize in (False, 'no'):
        opts.initialize = 0
    else:
        warnOpt("initialize", initialize)
    #
    if threads in (True, 'yes'):
        opts.threads = 1
    elif threads in (False, 'no'):
        opts.threads = 0
    else:
        warnOpt("threads", threads)
    #
    if thread_level == 'single':
        opts.thread_level = MPI_THREAD_SINGLE
    elif thread_level == 'funneled':
        opts.thread_level = MPI_THREAD_FUNNELED
    elif thread_level == 'serialized':
        opts.thread_level = MPI_THREAD_SERIALIZED
    elif thread_level == 'multiple':
        opts.thread_level = MPI_THREAD_MULTIPLE
    else:
        warnOpt("thread_level", thread_level)
    #
    if finalize is None:
        opts.finalize = opts.initialize
    elif finalize in (True, 'yes'):
        opts.finalize = 1
    elif finalize in (False, 'no'):
        opts.finalize = 0
    else:
        warnOpt("finalize", finalize)
    #
    if fast_reduce in (True, 'yes'):
        opts.fast_reduce = 1
    elif fast_reduce in (False, 'no'):
        opts.fast_reduce = 0
    else:
        warnOpt("fast_reduce", fast_reduce)
    #
    if recv_mprobe in (True, 'yes'):
        opts.recv_mprobe = 1 and USE_MATCHED_RECV
    elif recv_mprobe in (False, 'no'):
        opts.recv_mprobe = 0
    else:
        warnOpt("recv_mprobe", recv_mprobe)
    #
    if errors == 'default':
        opts.errors = 0
    elif errors == 'exception':
        opts.errors = 1
    elif errors == 'fatal':
        opts.errors = 2
    else:
        warnOpt("errors", errors)
    #
    return 0

# -----------------------------------------------------------------------------

cdef extern from *:
    int PyMPI_Commctx_finalize() nogil

cdef int bootstrap() except -1:
    # Get options from 'mpi4pyve.rc' module
    getOptions(&options)
    # Cleanup at (the very end of) Python exit
    if Py_AtExit(atexit) < 0:
        PySys_WriteStderr(b"warning: could not register "
                          b"cleanup with Py_AtExit()%s", b"\n")
    # Do we have to initialize MPI?
    cdef int initialized = 1
    <void>MPI_Initialized(&initialized)
    if initialized:
        options.finalize = 0
        return 0
    if not options.initialize:
        return 0
    # MPI initialization
    cdef int ierr = MPI_SUCCESS
    cdef int required = MPI_THREAD_SINGLE
    cdef int provided = MPI_THREAD_SINGLE
    if options.threads:
        required = options.thread_level
        ierr = MPI_Init_thread(NULL, NULL, required, &provided)
        if ierr != MPI_SUCCESS: raise RuntimeError(
            "MPI_Init_thread() failed [error code: %d]" % ierr)
    else:
        ierr = MPI_Init(NULL, NULL)
        if ierr != MPI_SUCCESS: raise RuntimeError(
            "MPI_Init() failed [error code: %d]" % ierr)
    return 0

cdef inline int mpi_active() nogil:
    cdef int ierr = MPI_SUCCESS
    # MPI initialized ?
    cdef int initialized = 0
    ierr = MPI_Initialized(&initialized)
    if not initialized or ierr != MPI_SUCCESS: return 0
    # MPI finalized ?
    cdef int finalized = 1
    ierr = MPI_Finalized(&finalized)
    if finalized or ierr != MPI_SUCCESS: return 0
    # MPI should be active ...
    return 1

cdef int initialize() nogil except -1:
    if not mpi_active(): return 0
    comm_set_eh(MPI_COMM_SELF)
    comm_set_eh(MPI_COMM_WORLD)
    return 0

cdef void finalize() nogil:
    if not mpi_active(): return
    <void>PyMPI_Commctx_finalize()

cdef int abort_status = 0

cdef void atexit() nogil:
    if not mpi_active(): return
    if abort_status:
        <void>MPI_Abort(MPI_COMM_WORLD, abort_status)
    finalize()
    if options.finalize:
        <void>MPI_Finalize()

def _set_abort_status(object status):
    "Helper for ``python -m mpi4pyve.run ...``"
    global abort_status
    try:
        abort_status = status
    except:
        abort_status = 1 if status else 0

def print_option():
    print('initialize   :', options.initialize)
    print('threads      :', options.threads)
    print('thread_level :', options.thread_level)
    print('finalize     :', options.finalize)
    print('fast_reduce  :', options.fast_reduce)
    print('recv_mprobe  :', options.recv_mprobe)
    print('errors       :', options.errors)

# -----------------------------------------------------------------------------

# Number of processes assigned to each VH when started with multiple VH.
import os
from libc.stdlib cimport malloc, free
from libc.string cimport strcmp

cdef int get_mpi_local_size_from_nodeid(int nodeid):
    cdef int local_size = 0
    cdef int size
    comm = MPI_COMM_WORLD
    MPI_Comm_size(comm, &size)
    cdef int* nodes_nodeid = <int*>malloc(sizeof(int) * size)
    MPI_Allgather(&nodeid, 1, MPI_INT, nodes_nodeid, 1, MPI_INT, comm)
    for rank in range(0, size):
        if nodeid ==  nodes_nodeid[rank]:
            local_size += 1
    free(nodes_nodeid)
    return local_size


cdef int get_mpi_local_size_from_processname():
    cdef int local_size = 0
    cdef char processor_name[MPI_MAX_PROCESSOR_NAME + 1]
    cdef int resultlen 
    cdef int size
    comm = MPI_COMM_WORLD
    MPI_Comm_size(comm, &size)
    MPI_Get_processor_name(processor_name, &resultlen)
    cdef char* nodes_processor_name = <char*>malloc(sizeof(processor_name) * size)
    MPI_Allgather(processor_name, sizeof(processor_name), MPI_CHAR, 
                  nodes_processor_name, sizeof(processor_name) ,MPI_CHAR, comm)
    for rank in range(0, size):
        if strcmp( processor_name , &nodes_processor_name[rank * sizeof(processor_name)]) == 0:
            local_size += 1
    free(nodes_processor_name)
    return local_size

cdef void set_mpi_local_size():
    if not mpi_active(): return
    os.environ["_MPI4PYVE_MPI_INITIALIZED"] = '1'

    cdef int nodeid = -1
    try:
        nodeid = int(os.environ['MPINODEID'])
    except:
        pass

    cdef int local_size = 0
    if nodeid >= 0:
        local_size = get_mpi_local_size_from_nodeid(nodeid)
    else:
        local_size = get_mpi_local_size_from_processname()
    os.environ["_MPI4PYVE_MPI_LOCAL_SIZE"] = str(local_size)
        

# -----------------------------------------------------------------------------

# Vile hack for raising a exception and not contaminate the traceback

cdef extern from *:
    enum: PyMPI_ERR_UNAVAILABLE

cdef extern from "Python.h":
    void PyErr_SetObject(object, object)
    void *PyExc_RuntimeError
    void *PyExc_NotImplementedError

cdef object MPIException = <object>PyExc_RuntimeError

cdef int PyMPI_Raise(int ierr) except -1 with gil:
    if ierr == PyMPI_ERR_UNAVAILABLE:
        PyErr_SetObject(<object>PyExc_NotImplementedError, None)
        return 0
    if (<void*>MPIException) != NULL:
        PyErr_SetObject(MPIException, <long>ierr)
    else:
        PyErr_SetObject(<object>PyExc_RuntimeError, <long>ierr)
    return 0

cdef inline int CHKERR(int ierr) nogil except -1:
    if ierr == MPI_SUCCESS: return 0
    PyMPI_Raise(ierr)
    return -1

cdef inline void print_traceback():
    cdef object sys, traceback
    import sys, traceback
    traceback.print_exc()
    try: sys.stderr.flush()
    except: pass

# -----------------------------------------------------------------------------

# PyPy: Py_IsInitialized() cannot be called without the GIL

cdef extern from "Python.h":
    int _Py_IsInitialized"Py_IsInitialized"() nogil

cdef object _pypy_sentinel = None

cdef inline int Py_IsInitialized() nogil:
    if PYPY and (<void*>_pypy_sentinel) == NULL: return 0
    return _Py_IsInitialized()

# -----------------------------------------------------------------------------
