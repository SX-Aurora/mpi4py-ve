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

cdef class Message:

    """
    Message
    """

    def __cinit__(self, Message message=None):
        self.ob_mpi = MPI_MESSAGE_NULL
        if message is None: return
        self.ob_mpi = message.ob_mpi
        self.ob_buf = message.ob_buf

    def __dealloc__(self):
        if not (self.flags & PyMPI_OWNED): return
        CHKERR( del_Message(&self.ob_mpi) )

    def __richcmp__(self, other, int op):
        if not isinstance(other, Message): return NotImplemented
        cdef Message s = <Message>self, o = <Message>other
        if   op == Py_EQ: return (s.ob_mpi == o.ob_mpi)
        elif op == Py_NE: return (s.ob_mpi != o.ob_mpi)
        cdef mod = type(self).__module__
        cdef cls = type(self).__name__
        raise TypeError("unorderable type: '%s.%s'" % (mod, cls))

    def __bool__(self):
        return self.ob_mpi != MPI_MESSAGE_NULL

    # Matching Probe
    # --------------

    @classmethod
    def Probe(cls, Comm comm,
              int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None):
        """
        Blocking test for a matched message
        """
        cdef MPI_Message cmessage = MPI_MESSAGE_NULL
        cdef MPI_Status *statusp = arg_Status(status)
        with nogil: CHKERR( MPI_Mprobe(
            source, tag, comm.ob_mpi, &cmessage, statusp) )
        cdef Message message = <Message>Message.__new__(cls)
        message.ob_mpi = cmessage
        return message

    @classmethod
    def Iprobe(cls, Comm comm,
               int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None):
        """
        Nonblocking test for a matched message
        """
        cdef int flag = 0
        cdef MPI_Message cmessage = MPI_MESSAGE_NULL
        cdef MPI_Status *statusp = arg_Status(status)
        with nogil: CHKERR( MPI_Improbe(
             source, tag, comm.ob_mpi, &flag, &cmessage, statusp) )
        if flag == 0: return None
        cdef Message message = <Message>Message.__new__(cls)
        message.ob_mpi = cmessage
        return message

    # Matched receives
    # ----------------

    def Recv(self, buf, Status status=None):
        """
        Blocking receive of matched message
        """
        cdef MPI_Message message = self.ob_mpi
        cdef int source = MPI_ANY_SOURCE
        if message == MPI_MESSAGE_NO_PROC:
            source = MPI_PROC_NULL
        cdef _p_msg_p2p rmsg = message_p2p_recv(buf, source)
        cdef MPI_Status *statusp = arg_Status(status)
        with nogil: CHKERR( MPI_Mrecv(
            rmsg.buf, rmsg.count, rmsg.dtype,
            &message, statusp) )
        if self is not __MESSAGE_NO_PROC__:
            self.ob_mpi = message

    def Irecv(self, buf):
        """
        Nonblocking receive of matched message
        """
        cdef MPI_Message message = self.ob_mpi
        cdef int source = MPI_ANY_SOURCE
        if message == MPI_MESSAGE_NO_PROC:
            source = MPI_PROC_NULL
        cdef _p_msg_p2p rmsg = message_p2p_recv(buf, source)
        cdef Request request = <Request>Request.__new__(Request)
        with nogil: CHKERR( MPI_Imrecv(
            rmsg.buf, rmsg.count, rmsg.dtype,
            &message, &request.ob_mpi) )
        if self is not __MESSAGE_NO_PROC__:
            self.ob_mpi = message
        request.ob_buf = rmsg
        return request

    # Python Communication
    # --------------------
    #
    @classmethod
    def probe(cls, Comm comm,
              int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None):
        """Blocking test for a matched message"""
        cdef Message message = <Message>Message.__new__(cls)
        cdef MPI_Status *statusp = arg_Status(status)
        message.ob_buf = PyMPI_mprobe(source, tag, comm.ob_mpi,
                                      &message.ob_mpi, statusp)
        return message
    #
    @classmethod
    def iprobe(cls, Comm comm,
               int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None):
        """Nonblocking test for a matched message"""
        cdef int flag = 0
        cdef Message message = <Message>Message.__new__(cls)
        cdef MPI_Status *statusp = arg_Status(status)
        message.ob_buf = PyMPI_improbe(source, tag, comm.ob_mpi, &flag,
                                       &message.ob_mpi, statusp)
        if flag == 0: return None
        return message
    #
    def recv(self, Status status=None):
        """Blocking receive of matched message"""
        cdef object rmsg = self.ob_buf
        cdef MPI_Message message = self.ob_mpi
        cdef MPI_Status *statusp = arg_Status(status)
        rmsg = PyMPI_mrecv(rmsg, &message, statusp)
        if self is not __MESSAGE_NO_PROC__: self.ob_mpi = message
        if self.ob_mpi == MPI_MESSAGE_NULL: self.ob_buf = None
        return rmsg
    #
    def irecv(self):
        """Nonblocking receive of matched message"""
        cdef object rmsg = self.ob_buf
        cdef MPI_Message message = self.ob_mpi
        cdef Request request = <Request>Request.__new__(Request)
        request.ob_buf = PyMPI_imrecv(rmsg, &message, &request.ob_mpi)
        if self is not __MESSAGE_NO_PROC__: self.ob_mpi = message
        if self.ob_mpi == MPI_MESSAGE_NULL: self.ob_buf = None
        return request

    # Fortran Handle
    # --------------

    def py2f(self):
        """
        """
        return MPI_Message_c2f(self.ob_mpi)

    @classmethod
    def f2py(cls, arg):
        """
        """
        cdef Message message = <Message>Message.__new__(Message)
        message.ob_mpi = MPI_Message_f2c(arg)
        return message


cdef Message __MESSAGE_NULL__    = new_Message ( MPI_MESSAGE_NULL    )
cdef Message __MESSAGE_NO_PROC__ = new_Message ( MPI_MESSAGE_NO_PROC )


# Predefined message handles
# --------------------------

MESSAGE_NULL    = __MESSAGE_NULL__    #: Null message handle
MESSAGE_NO_PROC = __MESSAGE_NO_PROC__ #: No-proc message handle
