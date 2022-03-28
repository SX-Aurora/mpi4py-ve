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

import numpy
import nlcpy
import mpi4pyve


def _replace_nlcpy_to_numpy(args):
    if args is None:
        return args
    _type = type(args)
    _args = list(args)
    for i, arg in enumerate(_args):
        if isinstance(arg, nlcpy.core.core.ndarray):
            _args[i] = numpy.asarray(arg)
        elif isinstance(arg, (list, tuple)):
            _args[i] = _replace_nlcpy_to_numpy(arg)
    return _type(_args)


def _replace_nlcpy_to_numpy_kwargs(kwargs):
    for k in kwargs.keys():
        if isinstance(kwargs[k], nlcpy.core.core.ndarray):
            kwargs[k] = numpy.asarray(kwargs[k])
        elif isinstance(kwargs[k], (list, tuple)):
            kwargs[k] = _replace_nlcpy_to_numpy(kwargs[k])
    return kwargs


def _undo_numpy_to_nlcpy(args):
    if args is None:
        return args
    _type = type(args)
    _args = list(args)
    for i, arg in enumerate(_args):
        if isinstance(arg, numpy.ndarray):
            _args[i] = nlcpy.asarray(arg)
        elif isinstance(arg, (list, tuple)):
            _args[i] = _undo_numpy_to_nlcpy(arg)
    return _type(_args)


def _undo_numpy_to_nlcpy_kwargs(kwargs):
    for k in kwargs.keys():
        if isinstance(kwargs[k], nlcpy.core.core.ndarray):
            kwargs[k] = nlcpy.asarray(kwargs[k])
        elif isinstance(kwargs[k], (list, tuple)):
            kwargs[k] = _undo_numpy_to_nlcpy(kwargs[k])
    return kwargs


def send_for_nlcpy_array(send_func):
    def _get_numpy_array_wrapper(*args, **kwargs):
        args = _replace_nlcpy_to_numpy(args)
        kwargs = _replace_nlcpy_to_numpy_kwargs(kwargs)
        return send_func(*args, **kwargs)
    return _get_numpy_array_wrapper


def recv_for_nlcpy_array(recv_func):
    def _get_nlcpy_array_wrapper(*args, **kwargs):
        result = recv_func(*args, **kwargs)
        if isinstance(result, (list, tuple)):
            result = _undo_numpy_to_nlcpy(result)
        elif isinstance(result, numpy.ndarray):
            result = nlcpy.asarray(result)
        return result
    return _get_nlcpy_array_wrapper


def nb_recv_for_nlcpy_array(arg_idx):
    def _nb_recv_for_nlcpy_array(recv_func):
        def _get_nlcpy_array_wrapper(*args, **kwargs):
            if len(args) > arg_idx:
                if isinstance(args[arg_idx], (list, tuple)):
                    if isinstance(args[arg_idx][0], nlcpy.core.core.ndarray):
                        kwargs["nlcpy_arr"] = args[arg_idx][0]
                        args = _replace_nlcpy_to_numpy(args)
                        kwargs["numpy_arr"] = args[arg_idx][0]
                elif isinstance(args[arg_idx], nlcpy.core.core.ndarray):
                    kwargs["nlcpy_arr"] = args[arg_idx]
                    args = _replace_nlcpy_to_numpy(args)
                    kwargs["numpy_arr"] = args[arg_idx]
            return recv_func(*args, **kwargs)
        return _get_nlcpy_array_wrapper
    return _nb_recv_for_nlcpy_array


def recv_buffer_for_nlcpy_array(arg_idx):
    def _recv_buffer_for_nlcpy_array(recv_func):
        def _get_nlcpy_array_wrapper(*args, **kwargs):
            tmp_array, tmp_buf, tmp_buftype = None, None, None
            if len(args) > arg_idx:
                if isinstance(args[arg_idx], (list, tuple)):
                    if isinstance(args[arg_idx][0], nlcpy.core.core.ndarray):
                        tmp_buftype = type(args[arg_idx])
                        tmp_buf = list(args[arg_idx])
                        tmp_array = args[arg_idx][0]
                elif isinstance(args[arg_idx], nlcpy.core.core.ndarray):
                    tmp_array = args[arg_idx]
            args = _replace_nlcpy_to_numpy(args)
            recv_func(*args, **kwargs)
            if tmp_array is not None:
                args = _undo_numpy_to_nlcpy(args)
                if tmp_buf is not None:
                    tmp_array[:len(args[arg_idx][0])] = args[arg_idx][0]
                    tmp_buf[0] = tmp_array
                    _args = list(args)
                    _args[arg_idx] = tmp_buftype(tmp_buf)
                    args = tuple(_args)
                else:
                    tmp_array[:len(args[arg_idx])] = args[arg_idx]
                    _args = list(args)
                    _args[arg_idx] = tmp_array
                    args = tuple(_args)
        return _get_nlcpy_array_wrapper
    return _recv_buffer_for_nlcpy_array


def sendrecv_buffer_for_nlcpy_array(arg_idx):
    def _sendrecv_buffer_for_nlcpy_array(sendrecv_func):
        def _get_nlcpy_array_wrapper(*args, **kwargs):
            tmp_send_array, tmp_send_buf, tmp_send_buftype = None, None, None
            tmp_recv_array, tmp_recv_buf, tmp_recv_buftype = None, None, None
            if len(args) > arg_idx[0]:
                if isinstance(args[arg_idx[0]], (list, tuple)):
                    if isinstance(args[arg_idx[0]][0], nlcpy.core.core.ndarray):
                        tmp_send_buftype = type(args[arg_idx[0]])
                        tmp_send_buf = list(args[arg_idx[0]])
                        tmp_send_array = args[arg_idx[0]][0]
                elif isinstance(args[arg_idx[0]], nlcpy.core.core.ndarray):
                    tmp_send_array = args[arg_idx[0]]
            if len(args) > arg_idx[1]:
                if isinstance(args[arg_idx[1]], (list, tuple)):
                    if isinstance(args[arg_idx[1]][0], nlcpy.core.core.ndarray):
                        tmp_recv_buftype = type(args[arg_idx[1]])
                        tmp_recv_buf = list(args[arg_idx[1]])
                        tmp_recv_array = args[arg_idx[1]][0]
                elif isinstance(args[arg_idx[1]], nlcpy.core.core.ndarray):
                    tmp_recv_array = args[arg_idx[1]]
            args = _replace_nlcpy_to_numpy(args)
            result = sendrecv_func(*args, **kwargs)
            if tmp_send_array is not None or tmp_recv_array is not None:
                if tmp_send_array is not None and tmp_send_buf is None:
                    tmp_send_array[:len(args[arg_idx[0]])] = args[arg_idx[0]]
                    tmp_send_buf = tmp_send_array
                elif tmp_send_array is not None:
                    tmp_send_array[:len(args[arg_idx[0]][0])] = args[arg_idx[0]][0]
                    tmp_send_buf[0] = tmp_send_array
                    tmp_send_buf = tmp_send_buftype(tmp_send_buf)
                if tmp_recv_array is not None and tmp_recv_buf is None:
                    tmp_recv_array[:len(args[arg_idx[1]])] = args[arg_idx[1]]
                    tmp_recv_buf = tmp_recv_array
                elif tmp_recv_array is not None:
                    tmp_recv_array[:len(args[arg_idx[1]][0])] = args[arg_idx[1]][0]
                    tmp_recv_buf[0] = tmp_recv_array
                    tmp_recv_buf = tmp_recv_buftype(tmp_recv_buf)
                args = tuple((args[0], tmp_send_buf, tmp_recv_buf))
            return result
        return _get_nlcpy_array_wrapper
    return _sendrecv_buffer_for_nlcpy_array


def nb_sendrecv_buffer_for_nlcpy_array(arg_idx):
    def _nb_sendrecv_buffer_for_nlcpy_array(recv_func):
        def _get_nlcpy_array_wrapper(*args, **kwargs):
            if (isinstance(args[arg_idx[0]], (list, tuple)) or
                (not isinstance(args[arg_idx[0]],
                 (list, tuple, nlcpy.core.core.ndarray)) and
                 ((args[arg_idx[0]] is None) or
                 (args[arg_idx[0]] == mpi4pyve.MPI.IN_PLACE)))) and\
               (isinstance(args[arg_idx[1]], (list, tuple)) or
                (not isinstance(args[arg_idx[1]],
                 (list, tuple, nlcpy.core.core.ndarray)) and
                 ((args[arg_idx[1]] is None) or
                 (args[arg_idx[1]] == mpi4pyve.MPI.IN_PLACE)))):
                kwargs["send_nlcpy_arr"] = args[arg_idx[0]][0]\
                    if (isinstance(args[arg_idx[0]], (list, tuple)) and
                        len(args[arg_idx[0]]) > 0 and
                        isinstance(args[arg_idx[0]][0],
                                   nlcpy.core.core.ndarray)) else None
                kwargs["recv_nlcpy_arr"] = args[arg_idx[1]][0]\
                    if (isinstance(args[arg_idx[1]], (list, tuple)) and
                        len(args[arg_idx[1]]) > 0 and
                        isinstance(args[arg_idx[1]][0],
                                   nlcpy.core.core.ndarray)) else None
                args = _replace_nlcpy_to_numpy(args)
                kwargs["send_numpy_arr"] = args[arg_idx[0]][0]\
                    if kwargs["send_nlcpy_arr"] is not None else None
                kwargs["recv_numpy_arr"] = args[arg_idx[1]][0]\
                    if kwargs["recv_nlcpy_arr"] is not None else None

            elif (isinstance(args[arg_idx[0]], (list, tuple)) and
                  isinstance(args[arg_idx[1]], nlcpy.core.core.ndarray)):
                kwargs["send_nlcpy_arr"] = args[arg_idx[0]][0]\
                    if isinstance(args[arg_idx[0]][0], nlcpy.core.core.ndarray) else None
                kwargs["recv_nlcpy_arr"] = args[arg_idx[1]]\
                    if isinstance(args[arg_idx[1]], nlcpy.core.core.ndarray) else None
                args = _replace_nlcpy_to_numpy(args)
                kwargs["send_numpy_arr"] = args[arg_idx[0]][0]\
                    if kwargs["send_nlcpy_arr"] is not None else None
                kwargs["recv_numpy_arr"] = args[arg_idx[1]]\
                    if kwargs["recv_nlcpy_arr"] is not None else None

            elif (isinstance(args[arg_idx[0]], nlcpy.core.core.ndarray) and
                  isinstance(args[arg_idx[1]], (list, tuple))):
                kwargs["send_nlcpy_arr"] = args[arg_idx[0]]\
                    if isinstance(args[arg_idx[0]], nlcpy.core.core.ndarray) else None
                kwargs["recv_nlcpy_arr"] = args[arg_idx[1]][0]\
                    if isinstance(args[arg_idx[1]][0], nlcpy.core.core.ndarray) else None
                args = _replace_nlcpy_to_numpy(args)
                kwargs["send_numpy_arr"] = args[arg_idx[0]]\
                    if kwargs["send_nlcpy_arr"] is not None else None
                kwargs["recv_numpy_arr"] = args[arg_idx[1]][0]\
                    if kwargs["recv_nlcpy_arr"] is not None else None

            elif (isinstance(args[arg_idx[0]], nlcpy.core.core.ndarray) or
                  args[arg_idx[0]] is None or
                  args[arg_idx[0]] == mpi4pyve.MPI.IN_PLACE) and\
                 (isinstance(args[arg_idx[1]], nlcpy.core.core.ndarray) or
                  args[arg_idx[1]] is None or args[arg_idx[1]] == mpi4pyve.MPI.IN_PLACE):
                kwargs["send_nlcpy_arr"] = args[arg_idx[0]]\
                    if isinstance(args[arg_idx[0]], nlcpy.core.core.ndarray) else None
                kwargs["recv_nlcpy_arr"] = args[arg_idx[1]]\
                    if isinstance(args[arg_idx[1]], nlcpy.core.core.ndarray) else None
                args = _replace_nlcpy_to_numpy(args)
                kwargs["send_numpy_arr"] = args[arg_idx[0]]\
                    if kwargs["send_nlcpy_arr"] is not None else None
                kwargs["recv_numpy_arr"] = args[arg_idx[1]]\
                    if kwargs["recv_nlcpy_arr"] is not None else None

            return recv_func(*args, **kwargs)
        return _get_nlcpy_array_wrapper
    return _nb_sendrecv_buffer_for_nlcpy_array


def sendrecv_buffer_kwarg_for_nlcpy_array(recv_func):
    def _get_nlcpy_array_wrapper(*args, **kwargs):
        tmp_send_array = None
        tmp_send_buf = None
        tmp_recv_array = None
        tmp_recv_buf = None
        sendbuf, sendkey = (kwargs['sendbuf'], True)\
            if 'sendbuf' in kwargs else (args[0], False)
        recvbuf, recvkey = (kwargs['recvbuf'], True)\
            if 'recvbuf' in kwargs else (args[4], False)\
            if len(args) > 4 else (None, False)

        if isinstance(sendbuf, list):
            if isinstance(sendbuf[0], nlcpy.core.core.ndarray):
                tmp_send_array = sendbuf[0]
                tmp_send_buf = sendbuf
        elif isinstance(sendbuf, nlcpy.core.core.ndarray):
            tmp_send_array = sendbuf

        if recvbuf is not None:
            if isinstance(recvbuf, list):
                if isinstance(recvbuf[0], nlcpy.core.core.ndarray):
                    tmp_recv_array = recvbuf[0]
                    tmp_recv_buf = recvbuf
            elif isinstance(recvbuf, nlcpy.core.core.ndarray):
                tmp_recv_array = recvbuf
        args = _replace_nlcpy_to_numpy(args)
        kwargs = _replace_nlcpy_to_numpy_kwargs(kwargs)

        recv_func(*args, **kwargs)

        sendfact = kwargs['sendbuf'] if sendkey else args[0]
        recvfact = kwargs['recvbuf'] if recvkey else args[4]\
            if len(args) > 4 else None
        args = _undo_numpy_to_nlcpy(args)
        kwargs = _undo_numpy_to_nlcpy_kwargs(kwargs)
        if tmp_send_array is not None:
            sendfact = tmp_send_array if tmp_send_buf is None else tmp_send_buf # NOQA
        if tmp_recv_array is not None:
            if tmp_recv_buf is not None:
                tmp_recv_array[:len(recvfact[0])] = recvfact[0]
                tmp_recv_buf[0] = tmp_recv_array
                recvfact = tmp_recv_buf
            else:
                tmp_recv_array[:len(recvfact)] = recvfact
                recvfact = tmp_recv_array
        return
    return _get_nlcpy_array_wrapper
