cdef inline int ve_is_contig(tuple shape,
                               tuple strides,
                               Py_ssize_t itemsize,
                               char order) except -1:
    cdef Py_ssize_t i, ndim = len(shape)
    cdef Py_ssize_t start, step, index
    if order == c'F':
        start = 0
        step = 1
    else:
        start = ndim - 1
        step = -1
    for i from 0 <= i < ndim:
        index = start + step * i
        if itemsize != <Py_ssize_t>strides[index]:
            return 0
        itemsize *= <Py_ssize_t>shape[index]
    return 1

cdef inline char* ve_get_format(char typekind, Py_ssize_t itemsize) nogil:
   if typekind == c'b':
       if itemsize == sizeof(char):      return b"?"
       if itemsize == (sizeof(char)*4):  return b"?" # NLCPy's bool is 4byte.
   if typekind == c'i':
       if itemsize == sizeof(char):      return b"b"
       if itemsize == sizeof(short):     return b"h"
       if itemsize == sizeof(int):       return b"i"
       if itemsize == sizeof(long):      return b"l"
       if itemsize == sizeof(long long): return b"q"
   if typekind == c'u':
       if itemsize == sizeof(char):      return b"B"
       if itemsize == sizeof(short):     return b"H"
       if itemsize == sizeof(int):       return b"I"
       if itemsize == sizeof(long):      return b"L"
       if itemsize == sizeof(long long): return b"Q"
   if typekind == c'f':
       if itemsize == sizeof(float)//2:    return b"e"
       if itemsize == sizeof(float):       return b"f"
       if itemsize == sizeof(double):      return b"d"
       if itemsize == sizeof(long double): return b"g"
   if typekind == c'c':
       if itemsize == 2*sizeof(float)//2:    return b"Ze"
       if itemsize == 2*sizeof(float):       return b"Zf"
       if itemsize == 2*sizeof(double):      return b"Zd"
       if itemsize == 2*sizeof(long double): return b"Zg"
   return BYTE_FMT

#------------------------------------------------------------------------------

cdef int Py_CheckVAIBuffer(object obj):
    try: return <bint>hasattr(obj, '__ve_array_interface__')
    except: return 0

cdef int Py_GetVAIBuffer(object obj, Py_buffer *view, int flags) except -1:
    cdef dict ve_array_interface
    cdef tuple data
    cdef str   typestr
    cdef tuple shape
    cdef tuple strides
    cdef list descr
    cdef object dev_ptr, mask
    cdef void *buf = NULL
    cdef bint readonly = 0
    cdef Py_ssize_t s, size = 1
    cdef Py_ssize_t itemsize = 1
    cdef char typekind = c'u'
    cdef bint fixnull = 0

    try:
        ve_array_interface = obj.__ve_array_interface__
    except AttributeError:
        raise NotImplementedError("missing VE array interface")

    # mandatory
    data = ve_array_interface['data']
    typestr = ve_array_interface['typestr']
    shape = tuple(ve_array_interface['shape'])

    # optional
    strides = None if ve_array_interface.get('strides') is None else \
        tuple(ve_array_interface.get('strides'))
    descr = ve_array_interface.get('descr')
    mask = ve_array_interface.get('mask')

    dev_ptr, readonly = data
    for s in shape: size *= s
    if dev_ptr is None and size == 0: dev_ptr = 0 # XXX
    buf = PyLong_AsVoidPtr(dev_ptr)
    typekind = <char>ord(typestr[1])
    itemsize = <Py_ssize_t>int(typestr[2:])

    if mask is not None:
        raise BufferError(
            "__ve_array_interface__: "
            "cannot handle masked arrays"
        )
    if size < 0:
        raise BufferError(
            "__ve_array_interface__: "
            "buffer with negative size (shape:%s, size:%d)"
            % (shape, size)
        )
    if (strides is not None and
        not ve_is_contig(shape, strides, itemsize, c'C') and
        not ve_is_contig(shape, strides, itemsize, c'F')):
        raise BufferError(
            "__ve_array_interface__: "
            "buffer is not contiguous (shape:%s, strides:%s, itemsize:%d)"
            % (shape, strides, itemsize)
        )
    if descr is not None and (len(descr) != 1 or descr[0] != ('', typestr)):
        PyErr_WarnEx(RuntimeWarning,
                     b"__ve_array_interface__: "
                     b"ignoring 'descr' key", 1)

    if PYPY and readonly and ((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE):
        raise BufferError("Object is not writable")

    fixnull = (buf == NULL and size == 0)
    if fixnull: buf = &fixnull
    PyBuffer_FillInfo(view, obj, buf, size*itemsize, readonly, flags)
    if fixnull: view.buf = NULL

    if (flags & PyBUF_FORMAT) == PyBUF_FORMAT:
        view.format = ve_get_format(typekind, itemsize)
        if view.format != BYTE_FMT:
            view.itemsize = itemsize
    return 0
