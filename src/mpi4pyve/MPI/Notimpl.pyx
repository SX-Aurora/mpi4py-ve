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

import nlcpy


def _find_nlcpy(args):
    for arg in args:
        _raise_nlcpy_array(arg)
    return


def _find_nlcpy_kwargs(kwargs):
    for k in kwargs.keys():
        _raise_nlcpy_array(kwargs[k])
    return


def _raise_nlcpy_array(arg):
    if isinstance(arg, nlcpy.core.core.ndarray):
        raise NotImplementedError('NLCPy array is not implemented yet.')
    elif isinstance(arg, (list, tuple)):
        _find_nlcpy(arg)
    elif isinstance(arg, dict):
        _find_nlcpy_kwargs(arg)
    return


def check_for_nlcpy_array(func):
    def _raise_nlcpy_array_wrapper(*args, **kwargs):
        _find_nlcpy(args)
        _find_nlcpy_kwargs(kwargs)
        return func(*args, **kwargs)
    return _raise_nlcpy_array_wrapper


def raise_notimpl_for_necmpi(func):
    def _raise_wrapper(*args, **kwargs):
        raise NotImplementedError('%s on mpi4py-ve is not implemented yet.'
                                  %func.__name__)
    return _raise_wrapper
