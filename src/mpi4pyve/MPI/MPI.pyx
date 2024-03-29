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

__doc__ = """
Message Passing Interface
"""

from mpi4pyve.libmpi cimport *

include "stdlib.pxi"
include "atimport.pxi"

bootstrap()
initialize()
set_mpi_local_size()

include "asstring.pxi"
include "asbuffer.pxi"
include "asmemory.pxi"
include "asarray.pxi"
include "helpers.pxi"
include "attrimpl.pxi"
include "mpierrhdl.pxi"
include "msgbuffer.pxi"
include "msgpickle.pxi"
include "CAPI.pxi"


# Assorted constants
# ------------------

UNDEFINED = MPI_UNDEFINED
#: Undefined integer value

ANY_SOURCE = MPI_ANY_SOURCE
#: Wildcard source value for receives

ANY_TAG = MPI_ANY_TAG
#: Wildcard tag value for receives

PROC_NULL = MPI_PROC_NULL
#: Special process rank for send/receive

ROOT = MPI_ROOT
#: Root process for collective inter-communications

BOTTOM = __BOTTOM__
#: Special address for buffers

IN_PLACE = __IN_PLACE__
#: *In-place* option for collective communications


# Predefined Attribute Keyvals
# ----------------------------

KEYVAL_INVALID    = MPI_KEYVAL_INVALID

TAG_UB            = MPI_TAG_UB
HOST              = MPI_HOST
IO                = MPI_IO
WTIME_IS_GLOBAL   = MPI_WTIME_IS_GLOBAL

UNIVERSE_SIZE     = MPI_UNIVERSE_SIZE
APPNUM            = MPI_APPNUM

LASTUSEDCODE      = MPI_LASTUSEDCODE

WIN_BASE          = MPI_WIN_BASE
WIN_SIZE          = MPI_WIN_SIZE
WIN_DISP_UNIT     = MPI_WIN_DISP_UNIT
WIN_CREATE_FLAVOR = MPI_WIN_CREATE_FLAVOR
WIN_FLAVOR        = MPI_WIN_CREATE_FLAVOR
WIN_MODEL         = MPI_WIN_MODEL


include "Exception.pyx"
include "Errhandler.pyx"
include "Notimpl.pyx"
include "Datatype.pyx"
include "Status.pyx"
include "Request.pyx"
include "Message.pyx"
include "Info.pyx"
include "Op.pyx"
include "Group.pyx"
include "Comm.pyx"
include "Win.pyx"
include "File.pyx"
include "Util.pyx"
include "Veo.pyx"


# Memory Allocation
# -----------------

def Alloc_mem(Aint size, Info info=INFO_NULL):
    """
    Allocate memory for message passing and RMA
    """
    cdef void *base = NULL
    CHKERR( MPI_Alloc_mem(size, info.ob_mpi, &base) )
    return tomemory(base, size)

def Free_mem(mem):
    """
    Free memory allocated with `Alloc_mem()`
    """
    cdef void *base = NULL
    cdef memory m = asmemory(mem, &base, NULL)
    CHKERR( MPI_Free_mem(base) )
    m.release()

# Initialization and Exit
# -----------------------

def Init():
    """
    Initialize the MPI execution environment
    """
    CHKERR( MPI_Init(NULL, NULL) )
    initialize()
    set_mpi_local_size()

def Finalize():
    """
    Terminate the MPI execution environment
    """
    finalize()
    CHKERR( MPI_Finalize() )

# Levels of MPI threading support
# -------------------------------

THREAD_SINGLE     = MPI_THREAD_SINGLE
#: Only one thread will execute

THREAD_FUNNELED   = MPI_THREAD_FUNNELED
#: MPI calls are *funneled* to the main thread

THREAD_SERIALIZED = MPI_THREAD_SERIALIZED
#: MPI calls are *serialized*

THREAD_MULTIPLE   = MPI_THREAD_MULTIPLE
#: Multiple threads may call MPI

def Init_thread(int required=THREAD_SERIALIZED):
    """
    Initialize the MPI execution environment
    """
    cdef int provided = MPI_THREAD_SINGLE

    if required == THREAD_MULTIPLE:
        PyErr_WarnEx(UserWarning, b"MPI_THREAD_MULTIPLE cannot be used with NEC MPI", 1)

    CHKERR( MPI_Init_thread(NULL, NULL, required, &provided) )
    initialize()
    set_mpi_local_size()
    return provided

def Query_thread():
    """
    Return the level of thread support
    provided by the MPI library
    """
    cdef int provided = MPI_THREAD_SINGLE
    CHKERR( MPI_Query_thread(&provided) )
    return provided

def Is_thread_main():
    """
    Indicate whether this thread called
    ``Init`` or ``Init_thread``
    """
    cdef int flag = 1
    CHKERR( MPI_Is_thread_main(&flag) )
    return <bint>flag

def Is_initialized():
    """
    Indicates whether ``Init`` has been called
    """
    cdef int flag = 0
    CHKERR( MPI_Initialized(&flag) )
    return <bint>flag

def Is_finalized():
    """
    Indicates whether ``Finalize`` has completed
    """
    cdef int flag = 0
    CHKERR( MPI_Finalized(&flag) )
    return <bint>flag

# Implementation Information
# --------------------------

# MPI Version Number
# -----------------

VERSION    = MPI_VERSION
SUBVERSION = MPI_SUBVERSION

def Get_version():
    """
    Obtain the version number of the MPI standard supported
    by the implementation as a tuple ``(version, subversion)``
    """
    cdef int version = 1
    cdef int subversion = 0
    CHKERR( MPI_Get_version(&version, &subversion) )
    return (version, subversion)

def Get_library_version():
    """
    Obtain the version string of the MPI library
    """
    cdef char name[MPI_MAX_LIBRARY_VERSION_STRING+1]
    cdef int nlen = 0
    CHKERR( MPI_Get_library_version(name, &nlen) )
    return tompistr(name, nlen)

# Environmental Inquires
# ----------------------

def Get_processor_name():
    """
    Obtain the name of the calling processor
    """
    cdef char name[MPI_MAX_PROCESSOR_NAME+1]
    cdef int nlen = 0
    CHKERR( MPI_Get_processor_name(name, &nlen) )
    return tompistr(name, nlen)

# Timers and Synchronization
# --------------------------

def Wtime():
    """
    Return an elapsed time on the calling processor
    """
    return MPI_Wtime()

def Wtick():
    """
    Return the resolution of ``Wtime``
    """
    return MPI_Wtick()

# Control of Profiling
# --------------------

def Pcontrol(int level):
    """
    Control profiling
    """
    if level < 0 or level > 2: CHKERR( MPI_ERR_ARG )
    CHKERR( MPI_Pcontrol(level) )


# Maximum string sizes
# --------------------

# MPI-1
MAX_PROCESSOR_NAME = MPI_MAX_PROCESSOR_NAME
MAX_ERROR_STRING   = MPI_MAX_ERROR_STRING
# MPI-2
MAX_PORT_NAME      = MPI_MAX_PORT_NAME
MAX_INFO_KEY       = MPI_MAX_INFO_KEY
MAX_INFO_VAL       = MPI_MAX_INFO_VAL
MAX_OBJECT_NAME    = MPI_MAX_OBJECT_NAME
MAX_DATAREP_STRING = MPI_MAX_DATAREP_STRING
# MPI-3
MAX_LIBRARY_VERSION_STRING = MPI_MAX_LIBRARY_VERSION_STRING

# --------------------------------------------------------------------

cdef extern from *:
    int PyMPI_Get_vendor(const char**,int*,int*,int*) nogil

def get_vendor():
    """
    Infomation about the underlying MPI implementation

    :Returns:
      - a string with the name of the MPI implementation
      - an integer 3-tuple version ``(major, minor, micro)``
    """
    cdef const char *name=NULL
    cdef int major=0, minor=0, micro=0
    CHKERR( PyMPI_Get_vendor(&name, &major, &minor, &micro) )
    return (mpistr(name), (major, minor, micro))

# --------------------------------------------------------------------

cdef extern from "Python.h":
    ctypedef ssize_t Py_intptr_t
    ctypedef size_t  Py_uintptr_t

cdef inline int _mpi_type(object arg, type cls) except -1:
    if isinstance(arg, type):
        if issubclass(arg, cls): return 1
    else:
        if isinstance(arg, cls): return 1
    return 0

def _sizeof(arg):
    """
    Size in bytes of the underlying MPI handle
    """
    if _mpi_type(arg, Status):     return sizeof(MPI_Status)
    if _mpi_type(arg, Datatype):   return sizeof(MPI_Datatype)
    if _mpi_type(arg, Request):    return sizeof(MPI_Request)
    if _mpi_type(arg, Message):    return sizeof(MPI_Message)
    if _mpi_type(arg, Op):         return sizeof(MPI_Op)
    if _mpi_type(arg, Group):      return sizeof(MPI_Group)
    if _mpi_type(arg, Info):       return sizeof(MPI_Info)
    if _mpi_type(arg, Errhandler): return sizeof(MPI_Errhandler)
    if _mpi_type(arg, Comm):       return sizeof(MPI_Comm)
    if _mpi_type(arg, Win):        return sizeof(MPI_Win)
    if _mpi_type(arg, File):       return sizeof(MPI_File)
    raise TypeError("expecting an MPI type or instance")

def _addressof(arg):
    """
    Memory address of the underlying MPI handle
    """
    cdef void *ptr = NULL
    if isinstance(arg, Status):
        ptr = <void*>&(<Status>arg).ob_mpi
    elif isinstance(arg, Datatype):
        ptr = <void*>&(<Datatype>arg).ob_mpi
    elif isinstance(arg, Request):
        ptr = <void*>&(<Request>arg).ob_mpi
    elif isinstance(arg, Message):
        ptr = <void*>&(<Message>arg).ob_mpi
    elif isinstance(arg, Op):
        ptr = <void*>&(<Op>arg).ob_mpi
    elif isinstance(arg, Group):
        ptr = <void*>&(<Group>arg).ob_mpi
    elif isinstance(arg, Info):
        ptr = <void*>&(<Info>arg).ob_mpi
    elif isinstance(arg, Errhandler):
        ptr = <void*>&(<Errhandler>arg).ob_mpi
    elif isinstance(arg, Comm):
        ptr = <void*>&(<Comm>arg).ob_mpi
    elif isinstance(arg, Win):
        ptr = <void*>&(<Win>arg).ob_mpi
    elif isinstance(arg, File):
        ptr = <void*>&(<File>arg).ob_mpi
    else:
        raise TypeError("expecting an MPI instance")
    return PyLong_FromVoidPtr(ptr)

def _handleof(arg):
    """
    Unsigned integer value with the underlying MPI handle
    """
    if isinstance(arg, Status):
        raise NotImplementedError
    elif isinstance(arg, Datatype):
        return <Py_uintptr_t>((<Datatype>arg).ob_mpi)
    elif isinstance(arg, Request):
        return <Py_uintptr_t>((<Request>arg).ob_mpi)
    elif isinstance(arg, Message):
        return <Py_uintptr_t>((<Message>arg).ob_mpi)
    elif isinstance(arg, Op):
        return <Py_uintptr_t>((<Op>arg).ob_mpi)
    elif isinstance(arg, Group):
        return <Py_uintptr_t>((<Group>arg).ob_mpi)
    elif isinstance(arg, Info):
        return <Py_uintptr_t>((<Info>arg).ob_mpi)
    elif isinstance(arg, Errhandler):
        return <Py_uintptr_t>((<Errhandler>arg).ob_mpi)
    elif isinstance(arg, Comm):
        return <Py_uintptr_t>((<Comm>arg).ob_mpi)
    elif isinstance(arg, Win):
        return <Py_uintptr_t>((<Win>arg).ob_mpi)
    elif isinstance(arg, File):
        return <Py_uintptr_t>((<File>arg).ob_mpi)
    else:
        raise TypeError("expecting an MPI instance")

# --------------------------------------------------------------------
