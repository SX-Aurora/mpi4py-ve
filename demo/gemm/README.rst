Performs the matrix-matrix operations:

    C = A * B

where A, B, and C are n by n matrices.

This example must be satisfied following conditions:

    1. int(sqrt(nproc)) * int(sqrt(nproc)) == nproc
    2. n % int(sqrt(nproc)) == 0

Note that this example is not fully optimized for SX-Aurora TSUBASA.
This is just only a prototype to demonstrate gemm on multi processes.

Issuing at the command line for VH::

    $ mpirun -veo -np 4 python gemm.py -dev vh -dtype float -n 10000
    {'dev': 'vh', 'dtype': 'float', 'n': 10000}
    elapsed: 4.292237043380737 [sec], GFLOPS: 465.9574901820241
    result OK

Issuing at the command line for VE::

    $ VE_NLCPY_NODELIST=0,1,2,3 mpirun -veo -np 4 python gemm.py -dev ve -dtype float -n 10000
    {'dev': 've', 'dtype': 'float', 'n': 10000}
    elapsed: 0.17874383926391602 [sec], TFLOPS: 11.189196831824741
    result OK
