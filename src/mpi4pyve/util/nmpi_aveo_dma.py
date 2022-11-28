from mpi4pyve import MPI

class nmpi_aveo_dma(object):
    def __init__(self):
        self.__count = None
        self.__size = None
        self.__time = None

    @property
    def count(self):
        self.__count = MPI._get_dma_count()
        return self.__count

    @property
    def size(self):
        self.__size = MPI._get_dma_size()
        return self.__size

    @property
    def time(self):
        self.__time = MPI._get_dma_time()
        return self.__time
    
    def clear(self):
        MPI._nmpi_aveo_dma_clear()

    def show_stats(self):
        (count, size, time) = (self.count, self.size, self.time)

        sta = []
        sta.append(['dma_count', str(count[0]), str(count[1]), str(count[2])])
        sta.append(['dma_size', str(size[0]), str(size[1]), str(size[2])])
        sta.append(['dma_time', str(time[0]), str(time[1]), str(time[2])])
        
        maxname = 0
        max_ve_ve = 0
        max_ve_vh = 0
        max_vh_ve = 0
        for val in sta:
            if maxname < len(val[0]):
                maxname = len(val[0])
            if max_ve_ve < len(val[1]):
                max_ve_ve = len(val[1])
            if max_ve_vh < len(val[2]):
                max_ve_vh = len(val[2])
            if max_vh_ve < len(val[3]):
                max_vh_ve = len(val[3])

        if len(sta) > 0:
            sp1 = max(10, maxname)
            sp2 = max(10, max_ve_ve)
            sp3 = max(10, max_ve_vh)
            sp4 = max(10, max_vh_ve)
            prval = "Info %s VE->VE %s VE->VH %s VH->VE %s" % (sp1*' ', sp2*' ', sp3*' ', sp4*' ')
            print(prval + "\n" + "-"*(sp1+4) + "  " + "-"*(sp2+6) + "  " + "-"*(sp3+6) + "  " + "-"*(sp4+6))

        for val in sta:
            print("%s %s %s %s %s %s %s %s" % (val[0], ' '*(sp1-len(val[0])+4),
                                            val[1], ' '*(sp2-len(val[1])+6),
                                            val[2], ' '*(sp3-len(val[2])+6),
                                            val[3], ' '*(sp4-len(val[3])+6)))

