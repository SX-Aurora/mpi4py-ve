import nlcpy

class nlcpy_ndarray_wrapper(nlcpy.ndarray):
    def __init__(self, shape, dtype=float, strides=None, order='C'):
        super().__init__(shape, dtype, strides, order)
        self.read_only_flag = False

    def set_read_only_flag(self, read_only_flag):
        self.read_only_flag = read_only_flag

    @property
    def __ve_array_interface__(self):
        vai = super().__ve_array_interface__
        vai['data'] = (vai['data'][0], self.read_only_flag)
        return vai

