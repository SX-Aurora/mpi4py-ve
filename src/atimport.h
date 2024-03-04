/* Author:  Lisandro Dalcin   */
/* Contact: dalcinl@gmail.com */

/* ------------------------------------------------------------------------- */

#include "Python.h"
#include "mpi.h"

#ifdef MPI4PYVE_NEC_MPI
#include <stdlib.h>
#define MPI_Init(argc, argv) \
    ((getenv("NMPI_LOCAL_RANK") != NULL) ? MPI_Init(argc, argv) : MPI_ERR_OTHER)
#define MPI_Init_thread(argc, argv, required, provided) \
    ((getenv("NMPI_LOCAL_RANK") != NULL) ? MPI_Init_thread(argc, argv, required, provided) : MPI_ERR_OTHER)
#define MPI_Finalize() \
    ((getenv("NMPI_LOCAL_RANK") != NULL) ? MPI_Finalize() : MPI_SUCCESS)
#endif /* MPI4PYVE_NEC_MPI */

/* ------------------------------------------------------------------------- */

#include "lib-mpi/config.h"
#include "lib-mpi/missing.h"
#include "lib-mpi/fallback.h"
#include "lib-mpi/compat.h"

#include "pympivendor.h"
#include "pympicommctx.h"

/* ------------------------------------------------------------------------- */

#include "pycompat.h"

#ifdef PYPY_VERSION
  #define PyMPI_RUNTIME_PYPY    1
  #define PyMPI_RUNTIME_CPYTHON 0
#else
  #define PyMPI_RUNTIME_PYPY    0
  #define PyMPI_RUNTIME_CPYTHON 1
#endif

/* ------------------------------------------------------------------------- */

#if !defined(PyMPI_USE_MATCHED_RECV)
  #if defined(PyMPI_HAVE_MPI_Mprobe) && \
      defined(PyMPI_HAVE_MPI_Mrecv)  && \
      MPI_VERSION >= 3
    #define PyMPI_USE_MATCHED_RECV 1
  #else
    #define PyMPI_USE_MATCHED_RECV 0
  #endif
#endif

/* ------------------------------------------------------------------------- */

/*
  Local variables:
  c-basic-offset: 2
  indent-tabs-mode: nil
  End:
*/
