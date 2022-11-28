cdef extern long nmpi_aveo_dma_count[3]
cdef extern long nmpi_aveo_dma_size[3]
cdef extern double nmpi_aveo_dma_time[3]

def _get_dma_count():
    return [<long>nmpi_aveo_dma_count[0],
            <long>nmpi_aveo_dma_count[1],
            <long>nmpi_aveo_dma_count[2],]

def _get_dma_size():
    return [<long>nmpi_aveo_dma_size[0],
            <long>nmpi_aveo_dma_size[1],
            <long>nmpi_aveo_dma_size[2],]

def _get_dma_time():
    return [<double>nmpi_aveo_dma_time[0],
            <double>nmpi_aveo_dma_time[1],
            <double>nmpi_aveo_dma_time[2],]

def _nmpi_aveo_dma_clear():
    for i in range(3):
        nmpi_aveo_dma_count[i] = 0
        nmpi_aveo_dma_size[i] = 0
        nmpi_aveo_dma_time[i] = 0

