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

"""
This is the **MPI for Python** package.

What is *MPI*?
==============

The *Message Passing Interface*, is a standardized and portable
message-passing system designed to function on a wide variety of
parallel computers. The standard defines the syntax and semantics of
library routines and allows users to write portable programs in the
main scientific programming languages (Fortran, C, or C++). Since
its release, the MPI specification has become the leading standard
for message-passing libraries for parallel computers.

What is *MPI for Python*?
=========================

*MPI for Python* provides MPI bindings for the Python programming
language, allowing any Python program to exploit multiple processors.
This package is constructed on top of the MPI-1/2 specifications and
provides an object oriented interface which closely follows MPI-2 C++
bindings.
"""

__version__ = '1.0.1'
__author__ = 'NEC (dev-nlcpy@sxarr.jp.nec.com)'
__credits__ = 'NEC Corporation'


__all__ = ['MPI']


def get_include():
    """Return the directory in the package that contains header files.

    Extension modules that need to compile against mpi4py-ve should use
    this function to locate the appropriate include directory. Using
    Python distutils (or perhaps NumPy distutils)::

      import mpi4pyve
      Extension('extension_name', ...
                include_dirs=[..., mpi4pyve.get_include()])

    """
    from os.path import join, dirname
    return join(dirname(__file__), 'include')


def get_config():
    """Return a dictionary with information about MPI."""
    from os.path import join, dirname
    try:
        from configparser import ConfigParser
    except ImportError:  # pragma: no cover
        from ConfigParser import ConfigParser
    parser = ConfigParser()
    parser.read(join(dirname(__file__), 'mpi.cfg'))
    return dict(parser.items('mpi'))


def rc(**kargs):  # pylint: disable=invalid-name
    """Runtime configuration options.

    Parameters
    ----------
    initialize : bool
        Automatic MPI initialization at import (default: True).
    threads : bool
        Request for thread support (default: True).
    thread_level : {'multiple', 'serialized', 'funneled', 'single'}
        Level of thread support to request (default: 'multiple').
    finalize : None or bool
        Automatic MPI finalization at exit (default: None).
    fast_reduce : bool
        Use tree-based reductions for objects (default: True).
    recv_mprobe : bool
        Use matched probes to receive objects (default: True).
    errors : {'exception', 'default', 'fatal'}
        Error handling policy (default: 'exception').

    """
    for key in kargs:
        if not hasattr(rc, key):
            raise TypeError("unexpected argument '{0}'".format(key))
    for key, value in kargs.items():
        setattr(rc, key, value)

rc.initialize = True
rc.threads = True
rc.thread_level = 'serialized'
rc.finalize = None
rc.fast_reduce = True
rc.recv_mprobe = True
rc.errors = 'exception'
__import__('sys').modules[__name__ + '.rc'] = rc


def profile(name, **kargs):
    """Support for the MPI profiling interface.

    Parameters
    ----------
    name : str
       Name of the profiler library to load.
    path : list of str, optional
       Additional paths to search for the profiler.
    logfile : str, optional
       Filename prefix for dumping profiler output.

    """
    import sys
    import os
    from .dl import dlopen, dlerror, RTLD_NOW, RTLD_GLOBAL

    def lookup_dylib(name, path):
        # pylint: disable=missing-docstring
        pattern = []
        if sys.platform.startswith('win'):  # pragma: no cover
            pattern.append(('', '.dll'))
        elif sys.platform == 'darwin':  # pragma: no cover
            pattern.append(('lib', '.dylib'))
        elif os.name == 'posix':  # pragma: no cover
            pattern.append(('lib', '.so'))
        pattern.append(('', ''))
        for pth in path:
            for (lib, dso) in pattern:
                filename = os.path.join(pth, lib + name + dso)
                if os.path.isfile(filename):
                    return os.path.abspath(filename)
        return None

    logfile = kargs.pop('logfile', None)
    if logfile:
        if name in ('mpe',):
            if 'MPE_LOGFILE_PREFIX' not in os.environ:
                os.environ['MPE_LOGFILE_PREFIX'] = logfile
        if name in ('vt', 'vt-mpi', 'vt-hyb'):
            if 'VT_FILE_PREFIX' not in os.environ:
                os.environ['VT_FILE_PREFIX'] = logfile

    path = kargs.pop('path', [])
    if isinstance(path, str):
        path = [path]
    else:
        path = list(path)
    prefix = os.path.dirname(__file__)
    path.append(os.path.join(prefix, 'lib-pmpi'))
    filename = lookup_dylib(name, path)
    if filename is None:
        raise ValueError("profiler '{0}' not found".format(name))

    handle = dlopen(filename, RTLD_NOW | RTLD_GLOBAL)
    if handle:
        profile.registry.append((name, (handle, filename)))
    else:
        from warnings import warn
        warn(dlerror())

profile.registry = []
