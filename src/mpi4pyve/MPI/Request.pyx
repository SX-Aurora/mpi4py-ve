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

def _replace_nlcpy_to_numpy(args):
    _type = type(args)
    _args = list(args)
    for i, arg in enumerate(_args):
        if isinstance(arg, nlcpy.core.core.ndarray):
            _args[i] = numpy.asarray(arg)
        elif isinstance(arg, (list, tuple)):
            _args[i] = _replace_nlcpy_to_numpy(arg)
    return _type(_args)


def _undo_numpy_to_nlcpy(args):
    _type = type(args)
    _args = list(args)
    for i, arg in enumerate(_args):
        if isinstance(arg, numpy.ndarray):
            _args[i] = nlcpy.asarray(arg)
        elif isinstance(arg, (list, tuple)):
            _args[i] = _undo_numpy_to_nlcpy(arg)
    return _type(_args)


cdef wait_for_nlcpy_array(wait_func):
    def _get_nlcpy_array_wrapper(*args, **kwargs):
        result = wait_func(*args, **kwargs)
        if isinstance(result, (list, tuple)):
            result = _undo_numpy_to_nlcpy(result)
        elif isinstance(result, numpy.ndarray):
            result = nlcpy.asarray(result)
        return result
    return _get_nlcpy_array_wrapper


def _get_numpy_to_nlcpy(args):
    if isinstance(args, (list, tuple)):
        return _undo_numpy_to_nlcpy(args)
    elif isinstance(args, numpy.ndarray):
        return nlcpy.asarray(args)
    else:
        return args


cdef class Request:

    """
    Request
    """

    def __cinit__(self, Request request=None, numpy_arr=None, nlcpy_arr=None):
        if numpy_arr is not None:
            self.numpy_arr = numpy_arr
            self.nlcpy_arr = nlcpy_arr

        self.ob_mpi = MPI_REQUEST_NULL
        if request is None: return
        self.ob_mpi = request.ob_mpi
        self.ob_buf = request.ob_buf

    def __dealloc__(self):
        if not (self.flags & PyMPI_OWNED): return
        CHKERR( del_Request(&self.ob_mpi) )

    def __richcmp__(self, other, int op):
        if not isinstance(other, Request): return NotImplemented
        cdef Request s = <Request>self, o = <Request>other
        if   op == Py_EQ: return (s.ob_mpi == o.ob_mpi)
        elif op == Py_NE: return (s.ob_mpi != o.ob_mpi)
        cdef mod = type(self).__module__
        cdef cls = type(self).__name__
        raise TypeError("unorderable type: '%s.%s'" % (mod, cls))

    def __bool__(self):
        return self.ob_mpi != MPI_REQUEST_NULL

    # Completion Operations
    # ---------------------

    def _trans_nlcpy_array(self):
        if self.numpy_arr is not None:
            if not isinstance(self.numpy_arr, list):
                self.nlcpy_arr[:] = self.numpy_arr
            else:
                for i in range(0, len(self.numpy_arr)):
                    if not self.numpy_arr[i] is None:
                        self.nlcpy_arr[i][:] = self.numpy_arr[i]
        return

    def Wait(self, Status status=None):
        """
        Wait for a send or receive to complete
        """

        cdef MPI_Status *statusp = arg_Status(status)
        with nogil: CHKERR( MPI_Wait(
            &self.ob_mpi, statusp) )
        if self.ob_mpi == MPI_REQUEST_NULL:
            self.ob_buf = None

        self._trans_nlcpy_array()
        return True

    def Test(self, Status status=None):
        """
        Test for the completion of a send or receive
        """
        cdef int flag = 0
        cdef MPI_Status *statusp = arg_Status(status)
        with nogil: CHKERR( MPI_Test(
            &self.ob_mpi, &flag, statusp) )
        if self.ob_mpi == MPI_REQUEST_NULL:
            self.ob_buf = None

        self._trans_nlcpy_array()
        return <bint>flag

    def Free(self):
        """
        Free a communication request
        """
        with nogil: CHKERR( MPI_Request_free(&self.ob_mpi) )

    def Get_status(self, Status status=None):
        """
        Non-destructive test for the completion of a request
        """
        cdef int flag = 0
        cdef MPI_Status *statusp = arg_Status(status)
        with nogil: CHKERR( MPI_Request_get_status(
            self.ob_mpi, &flag, statusp) )
        return <bint>flag

    # Multiple Completions
    # --------------------

    @classmethod
    def Waitany(cls, requests, Status status=None):
        """
        Wait for any previously initiated request to complete
        """
        cdef int count = 0
        cdef MPI_Request *irequests = NULL
        cdef int index = MPI_UNDEFINED
        cdef MPI_Status *statusp = arg_Status(status)
        #
        cdef tmp = acquire_rs(requests, None, &count, &irequests, NULL)
        try:
            with nogil: CHKERR( MPI_Waitany(
                count, irequests, &index, statusp) )
        finally:
            for req in requests:
                req._trans_nlcpy_array()
            release_rs(requests, None, count, irequests, NULL)
        return index

    @classmethod
    def Testany(cls, requests, Status status=None):
        """
        Test for completion of any previously initiated request
        """
        cdef int count = 0
        cdef MPI_Request *irequests = NULL
        cdef int index = MPI_UNDEFINED
        cdef int flag = 0
        cdef MPI_Status *statusp = arg_Status(status)
        #
        cdef tmp = acquire_rs(requests, None, &count, &irequests, NULL)
        try:
            with nogil: CHKERR( MPI_Testany(
                count, irequests, &index, &flag, statusp) )
        finally:
            for req in requests:
                req._trans_nlcpy_array()
            release_rs(requests, None, count, irequests, NULL)
        #
        return (index, <bint>flag)

    @classmethod
    def Waitall(cls, requests, statuses=None):
        """
        Wait for all previously initiated requests to complete
        """
        cdef int count = 0
        cdef MPI_Request *irequests = NULL
        cdef MPI_Status *istatuses = MPI_STATUSES_IGNORE
        #
        cdef tmp = acquire_rs(requests, statuses,
                              &count, &irequests, &istatuses)
        try:
            with nogil: CHKERR( MPI_Waitall(
                count, irequests, istatuses) )
        finally:
            for req in requests:
                req._trans_nlcpy_array()
            release_rs(requests, statuses, count, irequests, istatuses)
        return True

    @classmethod
    def Testall(cls, requests, statuses=None):
        """
        Test for completion of all previously initiated requests
        """
        cdef int count = 0
        cdef MPI_Request *irequests = NULL
        cdef int flag = 0
        cdef MPI_Status *istatuses = MPI_STATUSES_IGNORE
        #
        cdef tmp = acquire_rs(requests, statuses,
                              &count, &irequests, &istatuses)
        try:
            with nogil: CHKERR( MPI_Testall(
                count, irequests, &flag, istatuses) )
        finally:
            for req in requests:
                req._trans_nlcpy_array()
            release_rs(requests, statuses, count, irequests, istatuses)
        return <bint>flag

    @classmethod
    def Waitsome(cls, requests, statuses=None):
        """
        Wait for some previously initiated requests to complete
        """
        cdef int incount = 0
        cdef MPI_Request *irequests = NULL
        cdef int outcount = MPI_UNDEFINED, *iindices = NULL
        cdef MPI_Status *istatuses = MPI_STATUSES_IGNORE
        #
        cdef tmp1 = acquire_rs(requests, statuses,
                               &incount, &irequests, &istatuses)
        cdef tmp2 = newarray(incount, &iindices)
        try:
            with nogil: CHKERR( MPI_Waitsome(
                incount, irequests, &outcount, iindices, istatuses) )
        finally:
            for req in requests:
                req._trans_nlcpy_array()
            release_rs(requests, statuses, incount, irequests, istatuses)
        #
        cdef int i = 0
        cdef object indices = None
        if outcount != MPI_UNDEFINED:
            indices = [iindices[i] for i from 0 <= i < outcount]
        return indices

    @classmethod
    def Testsome(cls, requests, statuses=None):
        """
        Test for completion of some previously initiated requests
        """
        cdef int incount = 0
        cdef MPI_Request *irequests = NULL
        cdef int outcount = MPI_UNDEFINED, *iindices = NULL
        cdef MPI_Status *istatuses = MPI_STATUSES_IGNORE
        #
        cdef tmp1 = acquire_rs(requests, statuses,
                               &incount, &irequests, &istatuses)
        cdef tmp2 = newarray(incount, &iindices)
        try:
            with nogil: CHKERR( MPI_Testsome(
                incount, irequests, &outcount, iindices, istatuses) )
        finally:
            for req in requests:
                req._trans_nlcpy_array()
            release_rs(requests, statuses, incount, irequests, istatuses)
        #
        cdef int i = 0
        cdef object indices = None
        if outcount != MPI_UNDEFINED:
            indices = [iindices[i] for i from 0 <= i < outcount]
        return indices

    # Cancel
    # ------

    def Cancel(self):
        """
        Cancel a communication request
        """
        with nogil: CHKERR( MPI_Cancel(&self.ob_mpi) )

    # Fortran Handle
    # --------------

    def py2f(self):
        """
        """
        return MPI_Request_c2f(self.ob_mpi)

    @classmethod
    def f2py(cls, arg):
        """
        """
        cdef Request request = <Request>Request.__new__(Request)
        if issubclass(cls, Prequest):
            request = <Request>Prequest.__new__(Prequest)
        if issubclass(cls, Grequest):
            request = <Request>Grequest.__new__(Grequest)
        request.ob_mpi = MPI_Request_f2c(arg)
        return request

    # Python Communication
    # --------------------
    #
    @wait_for_nlcpy_array
    def wait(self, Status status=None):
        """
        Wait for a send or receive to complete
        """
        cdef msg = PyMPI_wait(self, status)
        self._trans_nlcpy_array()
        return msg
    #
    @wait_for_nlcpy_array
    def test(self, Status status=None):
        """
        Test for the completion of a send or receive
        """
        cdef int flag = 0
        cdef msg = PyMPI_test(self, &flag, status)
        self._trans_nlcpy_array()
        return (<bint>flag, msg)
    #
    @classmethod
    def waitany(cls, requests, Status status=None):
        """
        Wait for any previously initiated request to complete
        """
        cdef int index = MPI_UNDEFINED
        cdef msg = PyMPI_waitany(requests, &index, status)

        for req in requests:
            req._trans_nlcpy_array()
        msg = _get_numpy_to_nlcpy(msg)
        return (index, msg)
    #
    @classmethod
    def testany(cls, requests, Status status=None):
        """
        Test for completion of any previously initiated request
        """
        cdef int index = MPI_UNDEFINED
        cdef int flag  = 0
        cdef msg = PyMPI_testany(requests, &index, &flag, status)

        for req in requests:
            req._trans_nlcpy_array()
        msg = _get_numpy_to_nlcpy(msg)
        return (index, <bint>flag, msg)
    #
    @classmethod
    def waitall(cls, requests, statuses=None):
        """
        Wait for all previously initiated requests to complete
        """
        cdef msg = PyMPI_waitall(requests, statuses)

        for req in requests:
            req._trans_nlcpy_array()
        msg = _get_numpy_to_nlcpy(msg)
        return msg
    #
    @classmethod
    def testall(cls, requests, statuses=None):
        """
        Test for completion of all previously initiated requests
        """
        cdef int flag = 0
        cdef msg = PyMPI_testall(requests, &flag, statuses)

        for req in requests:
            req._trans_nlcpy_array()
        msg = _get_numpy_to_nlcpy(msg)
        return (<bint>flag, msg)


cdef class Prequest(Request):

    """
    Persistent request
    """

    def __cinit__(self, Request request=None, numpy_arr=None, nlcpy_arr=None):
        if numpy_arr is not None:
            self.numpy_arr = numpy_arr
            self.nlcpy_arr = nlcpy_arr

        if self.ob_mpi == MPI_REQUEST_NULL: return
        <void>(<Prequest?>request)

    def Start(self):
        """
        Initiate a communication with a persistent request
        """
        with nogil: CHKERR( MPI_Start(&self.ob_mpi) )

    @classmethod
    def Startall(cls, requests):
        """
        Start a collection of persistent requests
        """
        cdef int count = 0
        cdef MPI_Request *irequests = NULL
        cdef tmp = acquire_rs(requests, None, &count, &irequests, NULL)
        #
        try:
            with nogil: CHKERR( MPI_Startall(count, irequests) )
        finally:
            release_rs(requests, None, count, irequests, NULL)



cdef class Grequest(Request):

    """
    Generalized request
    """

    def __cinit__(self, Request request=None):
        self.ob_grequest = self.ob_mpi
        if self.ob_mpi == MPI_REQUEST_NULL: return
        <void>(<Grequest?>request)

    @classmethod
    def Start(cls, query_fn, free_fn, cancel_fn,
              args=None, kargs=None):
        """
        Create and return a user-defined request
        """
        cdef Grequest request = <Grequest>Grequest.__new__(Grequest)
        cdef _p_greq state = \
             _p_greq(query_fn, free_fn, cancel_fn, args, kargs)
        with nogil: CHKERR( MPI_Grequest_start(
            greq_query_fn, greq_free_fn, greq_cancel_fn,
            <void*>state, &request.ob_mpi) )
        Py_INCREF(state)
        request.ob_grequest = request.ob_mpi
        return request

    def Complete(self):
        """
        Notify that a user-defined request is complete
        """
        if self.ob_mpi != MPI_REQUEST_NULL:
            if self.ob_mpi != self.ob_grequest:
                raise MPIException(MPI_ERR_REQUEST)
        cdef MPI_Request grequest = self.ob_grequest
        self.ob_grequest = self.ob_mpi ## or MPI_REQUEST_NULL ??
        with nogil: CHKERR( MPI_Grequest_complete(grequest) )
        self.ob_grequest = self.ob_mpi ## or MPI_REQUEST_NULL ??



cdef Request __REQUEST_NULL__ = new_Request(MPI_REQUEST_NULL)


# Predefined request handles
# --------------------------

REQUEST_NULL = __REQUEST_NULL__  #: Null request handle
