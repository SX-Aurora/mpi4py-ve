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


from libc.stdint cimport *

cdef extern from "<dlfcn.h>" nogil:
    void *dlopen(const char *, int)
    char *dlerror()
    void *dlsym(void *, const char *)
    int dlclose(void *)
    int RTLD_LAZY
    int RTLD_NOW
    int RTLD_GLOBAL
    int RTLD_LOCAL

cdef int (*hooked_veo_alloc_hmem)(void *, void **, const size_t)
cdef int (*hooked_veo_free_hmem)(void *)


cdef _get_veo_sym():
    global hooked_veo_alloc_hmem, hooked_veo_free_hmem
    cdef void *hdl_veo = NULL
    cdef void *hdl_mpi = NULL
    cdef char *err = NULL

    hdl_veo = <void *>dlopen('libmpi_veo.so.1', RTLD_NOW)
    err = dlerror()
    if err != NULL:
        raise RuntimeError(err)
    hooked_veo_alloc_hmem = \
        <int (*)(void *, void **, const size_t)>dlsym(
            hdl_veo, 'veo_alloc_hmem')
    err = dlerror()
    if err != NULL:
        raise RuntimeError(err)
    hooked_veo_free_hmem = \
        <int (*)(void *)>dlsym(hdl_veo, 'veo_free_hmem')
    err = dlerror()
    if err != NULL:
        raise RuntimeError(err)

cdef int _hooked_alloc_hmem(void* proc, uint64_t* addr, const size_t size):
    global hooked_veo_alloc_hmem
    if hooked_veo_alloc_hmem == NULL:
        _get_veo_sym()
    cdef void *vemem = NULL
    cdef int ret = 0
    ret = hooked_veo_alloc_hmem(proc, &vemem, size)
    addr[0] = <uint64_t>vemem
    return ret

cdef int _hooked_free_hmem(uint64_t addr):
    global hooked_veo_free_hmem
    if hooked_veo_free_hmem == NULL:
        _get_veo_sym()
    cdef int ret = 0
    ret = hooked_veo_free_hmem(<void *>addr)
    return ret


def _alloc_hmem(uint64_t proc_handle, size_t size):
    cdef uint64_t hmem_addr = 0
    if _hooked_alloc_hmem(<void *>proc_handle, &hmem_addr, size):
        raise MemoryError("Out of memory on VE")
    return <uint64_t>(hmem_addr)


def _free_hmem(uint64_t addr):
    if _hooked_free_hmem(addr):
        raise RuntimeError("veo_free_hmem failed")
