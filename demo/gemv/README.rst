Performs the matrix-vector operations:

    y = A * x

where y is an m vector, x is an n vector, and A is an m by n matrix.

Issuing at the command line for VH::

    $ mpiexec -veo -np 4 python gemv.py -dev vh -dtype float -m 10000 -n 10000 -iter 100
    {'dev': 'vh', 'dtype': 'float', 'm': 10000, 'n': 10000, 'iter': 100}
    elapsed: 0.7454090118408203 [sec]
    Result success

Issuing at the command line for VE::

    $ VE_NLCPY_NODELIST=0,1,2,3 mpiexec -veo -np 4 python gemv.py -dev ve -dtype float -m 10000 -n 10000 -iter 100
    {'dev': 've', 'dtype': 'float', 'm': 10000, 'n': 10000, 'iter': 100}
    elapsed: 0.012457132339477539 [sec]
    Result success
