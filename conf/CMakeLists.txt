# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(mpi4pyve)

SET(PythonInterp_FIND_VERSION ${PYTHON_VERSION})
SET(PythonLibs_FIND_VERSION   ${PYTHON_VERSION_STRING})

FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(MPI)

SET(mpi4pyve_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
SET(mpi4pyve_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/mpi4pyve")

FILE(GLOB mpi4pyve_PYTHON_FILES
     RELATIVE ${mpi4pyve_SOURCE_DIR}
     ${mpi4pyve_SOURCE_DIR}/mpi4pyve/*.py
     ${mpi4pyve_SOURCE_DIR}/mpi4pyve/futures/*.py
     )

FILE(GLOB mpi4pyve_HEADER_FILES
     RELATIVE ${mpi4pyve_SOURCE_DIR}
     ${mpi4pyve_SOURCE_DIR}/mpi4pyve/*.pxd
     ${mpi4pyve_SOURCE_DIR}/mpi4pyve/include/mpi4pyve/*.[hi]
     ${mpi4pyve_SOURCE_DIR}/mpi4pyve/include/mpi4pyve/*.px[di]
    )

FOREACH(file
        ${mpi4pyve_PYTHON_FILES}
        ${mpi4pyve_HEADER_FILES}
        )
  SET(src "${mpi4pyve_SOURCE_DIR}/${file}")
  SET(tgt "${mpi4pyve_BINARY_DIR}/${file}")
  ADD_CUSTOM_COMMAND(
    DEPENDS ${src}
    OUTPUT  ${tgt}
    COMMAND ${CMAKE_COMMAND} ARGS -E copy ${src} ${tgt}
    COMMENT "copy: ${file}"
    )
  SET(mpi4pyve_OUTPUT_FILES ${mpi4pyve_OUTPUT_FILES} ${tgt})
ENDFOREACH(file)

##FOREACH(file ${mpi4pyve_PYTHON_FILES})
##  SET(mpi4pyve_py  ${mpi4pyve_py}  "${mpi4pyve_BINARY_DIR}/${file}")
##  SET(mpi4pyve_pyc ${mpi4pyve_pyc} "${mpi4pyve_BINARY_DIR}/${file}c")
##  SET(mpi4pyve_pyo ${mpi4pyve_pyo} "${mpi4pyve_BINARY_DIR}/${file}o")
##ENDFOREACH(file)
##ADD_CUSTOM_COMMAND(
##  COMMAND ${CMAKE_COMMAND} ARGS -E echo 'from compileall import compile_dir'     > compile_py
##  COMMAND ${CMAKE_COMMAND} ARGS -E echo 'compile_dir(\"${mpi4pyve_BINARY_DIR}\")' >> compile_py
##  COMMAND ${PYTHON_EXECUTABLE} ARGS compile_py
##  COMMAND ${PYTHON_EXECUTABLE} ARGS -O compile_py
##  COMMAND ${CMAKE_COMMAND} ARGS -E remove compile_py
##  DEPENDS ${mpi4pyve_py}
##  OUTPUT  ${mpi4pyve_pyc} ${mpi4pyve_pyo}
##  )
##SET(mpi4pyve_OUTPUT_FILES ${mpi4pyve_OUTPUT_FILES} ${mpi4pyve_pyc} ${mpi4pyve_pyo})


FIND_PROGRAM(MPI_COMPILER_CC
  NAMES mpicc
  HINTS "${MPI_BASE_DIR}"
  PATH_SUFFIXES bin
  DOC "MPI C compiler wrapper")
MARK_AS_ADVANCED(MPI_COMPILER_CC)
FIND_PROGRAM(MPI_COMPILER_CXX
  NAMES mpicxx mpic++ mpiCC
  HINTS "${MPI_BASE_DIR}"
  PATH_SUFFIXES bin
  DOC "MPI C++ compiler wrapper")
MARK_AS_ADVANCED(MPI_COMPILER_CXX)
find_program(MPI_COMPILER_FC
  NAMES mpifort mpif90 mpif77
  HINTS "${MPI_BASE_DIR}"
  PATH_SUFFIXES bin
  DOC "MPI Fortran compiler wrapper")
MARK_AS_ADVANCED(MPI_COMPILER_FC)
FIND_PROGRAM(MPI_COMPILER_F90
  NAMES mpif90
  HINTS "${MPI_BASE_DIR}"
  PATH_SUFFIXES bin
  DOC "MPI Fortran 90 compiler wrapper")
MARK_AS_ADVANCED(MPI_COMPILER_F90)
find_program(MPI_COMPILER_F77
  NAMES mpif77
  HINTS "${MPI_BASE_DIR}"
  PATH_SUFFIXES bin
  DOC "MPI Fortran 77 compiler wrapper")
MARK_AS_ADVANCED(MPI_COMPILER_F77)
FOREACH(file "mpi.cfg")
  SET(tgt "${mpi4pyve_BINARY_DIR}/${file}")
  ADD_CUSTOM_COMMAND(
    OUTPUT  ${tgt}
    COMMAND ${CMAKE_COMMAND} ARGS -E echo '[mpi]' > "${tgt}"
    COMMAND ${CMAKE_COMMAND} ARGS -E echo 'mpicc = ${MPI_COMPILER_CC}'   >> ${tgt}
    COMMAND ${CMAKE_COMMAND} ARGS -E echo 'mpicxx = ${MPI_COMPILER_CXX}' >> ${tgt}
    COMMAND ${CMAKE_COMMAND} ARGS -E echo 'mpifort = ${MPI_COMPILER_FC}' >> ${tgt}
    COMMAND ${CMAKE_COMMAND} ARGS -E echo 'mpif90 = ${MPI_COMPILER_F90}' >> ${tgt}
    COMMAND ${CMAKE_COMMAND} ARGS -E echo 'mpif77 = ${MPI_COMPILER_F77}' >> ${tgt}
    COMMENT "write: ${file}"
    )
  SET(mpi4pyve_OUTPUT_FILES ${mpi4pyve_OUTPUT_FILES} ${tgt})
ENDFOREACH(file)

ADD_CUSTOM_TARGET(mpi4pyve ALL DEPENDS ${mpi4pyve_OUTPUT_FILES})

INCLUDE_DIRECTORIES(
  ${MPI_INCLUDE_PATH}
  ${PYTHON_INCLUDE_PATH}
  "${mpi4pyve_SOURCE_DIR}"
  )

# --- mpi4pyve.MPI ---
PYTHON_ADD_MODULE(mpi4pyve.MPI MODULE "${mpi4pyve_SOURCE_DIR}/MPI.c")
SET_TARGET_PROPERTIES(
  mpi4pyve.MPI PROPERTIES
  OUTPUT_NAME "MPI" PREFIX ""
  COMPILE_FLAGS "${MPI_COMPILE_FLAGS}"
  LINK_FLAGS "${MPI_LINK_FLAGS}"
  LIBRARY_OUTPUT_DIRECTORY "${mpi4pyve_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${mpi4pyve_BINARY_DIR}"
  LINKER_LANGUAGE C
)
TARGET_LINK_LIBRARIES(mpi4pyve.MPI ${PYTHON_LIBRARY})
TARGET_LINK_LIBRARIES(mpi4pyve.MPI ${MPI_LIBRARIES})


# --- mpi4pyve.dl ---
PYTHON_ADD_MODULE(mpi4pyve.dl MODULE "${mpi4pyve_SOURCE_DIR}/dynload.c")
SET_TARGET_PROPERTIES(
  mpi4pyve.dl PROPERTIES
  OUTPUT_NAME "dl" PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${mpi4pyve_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${mpi4pyve_BINARY_DIR}"
  LINKER_LANGUAGE C
)
TARGET_LINK_LIBRARIES(mpi4pyve.dl ${PYTHON_LIBRARY})
TARGET_LINK_LIBRARIES(mpi4pyve.dl ${CMAKE_DL_LIBS})


# --- mpi4pyve/bin/python-mpi ---
ADD_EXECUTABLE(python-mpi "${mpi4pyve_SOURCE_DIR}/python.c")
SET_TARGET_PROPERTIES(
  python-mpi PROPERTIES
  COMPILE_FLAGS "${MPI_COMPILE_FLAGS}"
  LINK_FLAGS "${MPI_LINK_FLAGS}"
  RUNTIME_OUTPUT_DIRECTORY "${mpi4pyve_BINARY_DIR}/bin"
  LINKER_LANGUAGE C
)
TARGET_LINK_LIBRARIES(python-mpi ${PYTHON_LIBRARY})
TARGET_LINK_LIBRARIES(python-mpi ${MPI_LIBRARIES})

# --- mpi4pyve/lib-pmpi/libmpe.so ---
ADD_LIBRARY(pmpi-mpe MODULE "${mpi4pyve_SOURCE_DIR}/lib-pmpi/mpe.c")
SET_TARGET_PROPERTIES(
  pmpi-mpe PROPERTIES
  OUTPUT_NAME "mpe"
  LIBRARY_OUTPUT_DIRECTORY "${mpi4pyve_BINARY_DIR}/lib-pmpi"
  RUNTIME_OUTPUT_DIRECTORY "${mpi4pyve_BINARY_DIR}/lib-pmpi"
  LINKER_LANGUAGE C
)
TARGET_LINK_LIBRARIES(pmpi-mpe ${MPE_LIBRARIES})
TARGET_LINK_LIBRARIES(pmpi-mpe ${MPI_LIBRARIES})

# --- mpi4pyve/lib-pmpi/libvt.so ---
ADD_LIBRARY(pmpi-vt MODULE "${mpi4pyve_SOURCE_DIR}/lib-pmpi/vt.c")
SET_TARGET_PROPERTIES(
  pmpi-vt PROPERTIES
  OUTPUT_NAME "vt"
  LIBRARY_OUTPUT_DIRECTORY "${mpi4pyve_BINARY_DIR}/lib-pmpi"
  RUNTIME_OUTPUT_DIRECTORY "${mpi4pyve_BINARY_DIR}/lib-pmpi"
  LINKER_LANGUAGE C
)
TARGET_LINK_LIBRARIES(pmpi-vt ${VT_LIBRARIES})
TARGET_LINK_LIBRARIES(pmpi-vt ${MPI_LIBRARIES})

# --- mpi4pyve/lib-pmpi/libvt-mpi.so ---
ADD_LIBRARY(pmpi-vt-mpi MODULE "${mpi4pyve_SOURCE_DIR}/lib-pmpi/vt-mpi.c")
SET_TARGET_PROPERTIES(
  pmpi-vt-mpi PROPERTIES
  OUTPUT_NAME "vt-mpi"
  LIBRARY_OUTPUT_DIRECTORY "${mpi4pyve_BINARY_DIR}/lib-pmpi"
  RUNTIME_OUTPUT_DIRECTORY "${mpi4pyve_BINARY_DIR}/lib-pmpi"
  LINKER_LANGUAGE C
)
TARGET_LINK_LIBRARIES(pmpi-vt-mpi ${VT_MPI_LIBRARIES})
TARGET_LINK_LIBRARIES(pmpi-vt-mpi ${MPI_LIBRARIES})

# --- mpi4pyve/lib-pmpi/libvt-hyb.so ---
ADD_LIBRARY(pmpi-vt-hyb MODULE "${mpi4pyve_SOURCE_DIR}/lib-pmpi/vt-hyb.c")
SET_TARGET_PROPERTIES(
  pmpi-vt-hyb PROPERTIES
  OUTPUT_NAME "vt-hyb"
  LIBRARY_OUTPUT_DIRECTORY "${mpi4pyve_BINARY_DIR}/lib-pmpi"
  RUNTIME_OUTPUT_DIRECTORY "${mpi4pyve_BINARY_DIR}/lib-pmpi"
  LINKER_LANGUAGE C
)
TARGET_LINK_LIBRARIES(pmpi-vt-hyb ${VT_HYB_LIBRARIES})
TARGET_LINK_LIBRARIES(pmpi-vt-hyb ${MPI_LIBRARIES})
