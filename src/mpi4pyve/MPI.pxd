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

from mpi4pyve.libmpi cimport MPI_Aint
from mpi4pyve.libmpi cimport MPI_Offset
from mpi4pyve.libmpi cimport MPI_Count
from mpi4pyve.libmpi cimport MPI_Status
from mpi4pyve.libmpi cimport MPI_Datatype
from mpi4pyve.libmpi cimport MPI_Request
from mpi4pyve.libmpi cimport MPI_Message
from mpi4pyve.libmpi cimport MPI_Op
from mpi4pyve.libmpi cimport MPI_Group
from mpi4pyve.libmpi cimport MPI_Info
from mpi4pyve.libmpi cimport MPI_Errhandler
from mpi4pyve.libmpi cimport MPI_Comm
from mpi4pyve.libmpi cimport MPI_Win
from mpi4pyve.libmpi cimport MPI_File

# --

cdef import from *:
    ctypedef MPI_Aint   Aint   "MPI_Aint"
    ctypedef MPI_Offset Offset "MPI_Offset"
    ctypedef MPI_Count  Count  "MPI_Count"

ctypedef public api class Status [
    type   PyMPIStatus_Type,
    object PyMPIStatusObject,
    ]:
    cdef MPI_Status ob_mpi
    cdef unsigned   flags

ctypedef public api class Datatype [
    type   PyMPIDatatype_Type,
    object PyMPIDatatypeObject,
    ]:
    cdef MPI_Datatype ob_mpi
    cdef unsigned     flags

ctypedef public api class Request [
    type   PyMPIRequest_Type,
    object PyMPIRequestObject,
    ]:
    cdef MPI_Request ob_mpi
    cdef unsigned    flags
    cdef object      ob_buf

ctypedef public api class Prequest(Request) [
    type   PyMPIPrequest_Type,
    object PyMPIPrequestObject,
    ]:
    pass

ctypedef public api class Grequest(Request) [
    type   PyMPIGrequest_Type,
    object PyMPIGrequestObject,
    ]:
    cdef MPI_Request ob_grequest

ctypedef public api class Message [
    type   PyMPIMessage_Type,
    object PyMPIMessageObject,
    ]:
    cdef MPI_Message ob_mpi
    cdef unsigned    flags
    cdef object      ob_buf

ctypedef public api class Op [
    type   PyMPIOp_Type,
    object PyMPIOpObject,
    ]:
    cdef MPI_Op   ob_mpi
    cdef unsigned flags
    cdef object   (*ob_func)(object, object)
    cdef int      ob_usrid

ctypedef public api class Group [
    type   PyMPIGroup_Type,
    object PyMPIGroupObject,
    ]:
    cdef MPI_Group ob_mpi
    cdef unsigned  flags

ctypedef public api class Info [
    type   PyMPIInfo_Type,
    object PyMPIInfoObject,
    ]:
    cdef MPI_Info ob_mpi
    cdef unsigned flags

ctypedef public api class Errhandler [
    type   PyMPIErrhandler_Type,
    object PyMPIErrhandlerObject,
    ]:
    cdef MPI_Errhandler ob_mpi
    cdef unsigned       flags

ctypedef public api class Comm [
    type   PyMPIComm_Type,
    object PyMPICommObject,
    ]:
    cdef MPI_Comm ob_mpi
    cdef unsigned flags

ctypedef public api class Intracomm(Comm) [
    type   PyMPIIntracomm_Type,
    object PyMPIIntracommObject,
    ]:
    pass

ctypedef public api class Topocomm(Intracomm) [
    type   PyMPITopocomm_Type,
    object PyMPITopocommObject,
    ]:
    pass

ctypedef public api class Cartcomm(Topocomm) [
    type   PyMPICartcomm_Type,
    object PyMPICartcommObject,
    ]:
    pass

ctypedef public api class Graphcomm(Topocomm) [
    type   PyMPIGraphcomm_Type,
    object PyMPIGraphcommObject,
    ]:
    pass

ctypedef public api class Distgraphcomm(Topocomm) [
    type   PyMPIDistgraphcomm_Type,
    object PyMPIDistgraphcommObject,
    ]:
    pass

ctypedef public api class Intercomm(Comm) [
    type   PyMPIIntercomm_Type,
    object PyMPIIntercommObject,
    ]:
    pass

ctypedef public api class Win [
    type   PyMPIWin_Type,
    object PyMPIWinObject,
    ]:
    cdef MPI_Win  ob_mpi
    cdef unsigned flags
    cdef object   ob_mem

ctypedef public api class File [
    type   PyMPIFile_Type,
    object PyMPIFileObject,
    ]:
    cdef MPI_File ob_mpi
    cdef unsigned flags

# --
