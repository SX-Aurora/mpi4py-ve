/*
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
*/

/* ---------------------------------------------------------------- */

#if SWIG_VERSION < 0x010328
%warn "SWIG version < 1.3.28 is not supported"
#endif

/* ---------------------------------------------------------------- */

%header %{
#include "mpi4pyve/mpi4pyve.h"
%}

%init %{
if (import_mpi4pyve() < 0)
#if PY_MAJOR_VERSION >= 3
  return NULL;
#else
  return;
#endif
%}

/* ---------------------------------------------------------------- */

%define %mpi4pyve_fragments(PyType, Type)
/* --- AsPtr --- */
%fragment(SWIG_AsPtr_frag(Type),"header") {
SWIGINTERN int
SWIG_AsPtr_dec(Type)(SWIG_Object input, Type **p) {
  if (input == Py_None) {
    if (p) *p = NULL;
    return SWIG_OK;
  } else if (PyObject_TypeCheck(input,&PyMPI##PyType##_Type)) {
    if (p) *p = PyMPI##PyType##_Get(input);
    return SWIG_OK;
  } else {
    void *argp = NULL;
    int res = SWIG_ConvertPtr(input,&argp,%descriptor(p_##Type), 0);
    if (!SWIG_IsOK(res)) return res;
    if (!argp) return SWIG_ValueError;
    if (p) *p = %static_cast(argp,Type*);
    return SWIG_OK;
  }
}
}
/* --- From --- */
%fragment(SWIG_From_frag(Type),"header")
{
SWIGINTERN SWIG_Object
SWIG_From_dec(Type)(Type v) {
  return PyMPI##PyType##_New(v);
}
}
%enddef /*mpi4pyve_fragments*/

/* ---------------------------------------------------------------- */

%define SWIG_TYPECHECK_MPI_Comm       400 %enddef
%define SWIG_TYPECHECK_MPI_Datatype   401 %enddef
%define SWIG_TYPECHECK_MPI_Request    402 %enddef
%define SWIG_TYPECHECK_MPI_Message    403 %enddef
%define SWIG_TYPECHECK_MPI_Status     404 %enddef
%define SWIG_TYPECHECK_MPI_Op         405 %enddef
%define SWIG_TYPECHECK_MPI_Group      406 %enddef
%define SWIG_TYPECHECK_MPI_Info       407 %enddef
%define SWIG_TYPECHECK_MPI_File       408 %enddef
%define SWIG_TYPECHECK_MPI_Win        409 %enddef
%define SWIG_TYPECHECK_MPI_Errhandler 410 %enddef

/* ---------------------------------------------------------------- */

%define %mpi4pyve_typemap(PyType, Type)
%types(Type*);
%mpi4pyve_fragments(PyType, Type);
%typemaps_asptrfromn(%checkcode(Type), Type);
%enddef /*mpi4pyve_typemap*/

/* ---------------------------------------------------------------- */


/*
 * Local Variables:
 * mode: C
 * End:
 */
