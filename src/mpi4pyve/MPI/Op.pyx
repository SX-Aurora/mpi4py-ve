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

cdef class Op:

    """
    Op
    """

    def __cinit__(self, Op op=None):
        self.ob_mpi = MPI_OP_NULL
        if op is None: return
        self.ob_mpi = op.ob_mpi
        self.ob_func = op.ob_func
        self.ob_usrid = op.ob_usrid

    def __dealloc__(self):
        if not (self.flags & PyMPI_OWNED): return
        CHKERR( del_Op(&self.ob_mpi) )
        op_user_del(&self.ob_usrid)

    def __richcmp__(self, other, int op):
        if not isinstance(other, Op): return NotImplemented
        cdef Op s = <Op>self, o = <Op>other
        if   op == Py_EQ: return (s.ob_mpi == o.ob_mpi)
        elif op == Py_NE: return (s.ob_mpi != o.ob_mpi)
        cdef mod = type(self).__module__
        cdef cls = type(self).__name__
        raise TypeError("unorderable type: '%s.%s'" % (mod, cls))

    def __bool__(self):
        return self.ob_mpi != MPI_OP_NULL

    def __call__(self, x, y):
        if self.ob_func != NULL:
            return self.ob_func(x, y)
        else:
            return op_user_py(self.ob_usrid, x, y, None)

    @classmethod
    def Create(cls, function, bint commute=False):
        """
        Create a user-defined operation
        """
        cdef Op op = <Op>Op.__new__(Op)
        cdef MPI_User_function *cfunction = NULL
        op.ob_usrid = op_user_new(function, &cfunction)
        CHKERR( MPI_Op_create(cfunction, commute, &op.ob_mpi) )
        return op

    def Free(self):
        """
        Free the operation
        """
        CHKERR( MPI_Op_free(&self.ob_mpi) )
        op_user_del(&self.ob_usrid)
        if   self is __MAX__     : self.ob_mpi =  MPI_MAX
        elif self is __MIN__     : self.ob_mpi =  MPI_MIN
        elif self is __SUM__     : self.ob_mpi =  MPI_SUM
        elif self is __PROD__    : self.ob_mpi =  MPI_PROD
        elif self is __LAND__    : self.ob_mpi =  MPI_LAND
        elif self is __BAND__    : self.ob_mpi =  MPI_BAND
        elif self is __LOR__     : self.ob_mpi =  MPI_LOR
        elif self is __BOR__     : self.ob_mpi =  MPI_BOR
        elif self is __LXOR__    : self.ob_mpi =  MPI_LXOR
        elif self is __BXOR__    : self.ob_mpi =  MPI_BXOR
        elif self is __MAXLOC__  : self.ob_mpi =  MPI_MAXLOC
        elif self is __MINLOC__  : self.ob_mpi =  MPI_MINLOC
        elif self is __REPLACE__ : self.ob_mpi =  MPI_REPLACE
        elif self is __NO_OP__   : self.ob_mpi =  MPI_NO_OP

    # Process-local reduction
    # -----------------------

    def Is_commutative(self):
        """
        Query reduction operations for their commutativity
        """
        cdef int flag = 0
        CHKERR( MPI_Op_commutative(self.ob_mpi, &flag) )
        return <bint>flag

    property is_commutative:
        """is commutative"""
        def __get__(self):
            return self.Is_commutative()

    @raise_notimpl_for_vai_buffer
    def Reduce_local(self, inbuf, inoutbuf):
        """
        Apply a reduction operator to local data
        """
        # get *in* and *inout* buffers
        cdef _p_msg_cco m = message_cco()
        m.for_cro_send(inbuf, 0)
        m.for_cro_recv(inoutbuf, 0)
        # check counts and datatypes
        if m.scount != m.rcount: raise ValueError(
            "mismatch in inbuf count %d and inoutbuf count %d" %
            (m.scount, m.rcount))
        if (m.stype != m.rtype): raise ValueError(
            "mismatch in inbuf and inoutbuf MPI datatypes")
        # do local reduction
        with nogil: CHKERR( MPI_Reduce_local(
            m.sbuf, m.rbuf, m.rcount, m.rtype, self.ob_mpi) )

    property is_predefined:
        """is a predefined operation"""
        def __get__(self):
            cdef MPI_Op op = self.ob_mpi
            return (op == MPI_OP_NULL or
                    op == MPI_MAX or
                    op == MPI_MIN or
                    op == MPI_SUM or
                    op == MPI_PROD or
                    op == MPI_LAND or
                    op == MPI_BAND or
                    op == MPI_LOR or
                    op == MPI_BOR or
                    op == MPI_LXOR or
                    op == MPI_BXOR or
                    op == MPI_MAXLOC or
                    op == MPI_MINLOC or
                    op == MPI_REPLACE or
                    op == MPI_NO_OP)

    # Fortran Handle
    # --------------

    def py2f(self):
        """
        """
        return MPI_Op_c2f(self.ob_mpi)

    @classmethod
    def f2py(cls, arg):
        """
        """
        cdef Op op = <Op>Op.__new__(Op)
        op.ob_mpi = MPI_Op_f2c(arg)
        return op



cdef Op __OP_NULL__ = new_Op( MPI_OP_NULL )
cdef Op __MAX__     = new_Op( MPI_MAX     )
cdef Op __MIN__     = new_Op( MPI_MIN     )
cdef Op __SUM__     = new_Op( MPI_SUM     )
cdef Op __PROD__    = new_Op( MPI_PROD    )
cdef Op __LAND__    = new_Op( MPI_LAND    )
cdef Op __BAND__    = new_Op( MPI_BAND    )
cdef Op __LOR__     = new_Op( MPI_LOR     )
cdef Op __BOR__     = new_Op( MPI_BOR     )
cdef Op __LXOR__    = new_Op( MPI_LXOR    )
cdef Op __BXOR__    = new_Op( MPI_BXOR    )
cdef Op __MAXLOC__  = new_Op( MPI_MAXLOC  )
cdef Op __MINLOC__  = new_Op( MPI_MINLOC  )
cdef Op __REPLACE__ = new_Op( MPI_REPLACE )
cdef Op __NO_OP__   = new_Op( MPI_NO_OP   )


# Predefined operation handles
# ----------------------------

OP_NULL = __OP_NULL__  #: Null
MAX     = __MAX__      #: Maximum
MIN     = __MIN__      #: Minimum
SUM     = __SUM__      #: Sum
PROD    = __PROD__     #: Product
LAND    = __LAND__     #: Logical and
BAND    = __BAND__     #: Bit-wise and
LOR     = __LOR__      #: Logical or
BOR     = __BOR__      #: Bit-wise or
LXOR    = __LXOR__     #: Logical xor
BXOR    = __BXOR__     #: Bit-wise xor
MAXLOC  = __MAXLOC__   #: Maximum and location
MINLOC  = __MINLOC__   #: Minimum and location
REPLACE = __REPLACE__  #: Replace (for RMA)
NO_OP   = __NO_OP__    #: No-op   (for RMA)
