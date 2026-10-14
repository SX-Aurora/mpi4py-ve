#!/bin/sh

export NOSE_PARAMETERIZED_NO_WARN=1

pwd | grep -q "mpi4pyverepos$"
if [ $? != 0 ]; then
    echo "Error:Run this script in mpi4pyverepos directory." >&2
    exit 1
fi

module_enumeration()
{
    python -P -c "help('modules')" | grep -q "mpi4pyve"

  if [ $? != 0 ]; then
      echo "Error:mpi4pyve is not imported." >&2
      return 1
  fi

  return 0
}

module_enumeration

if [ $? = 0 ]; then
  echo "modules enumeration test passed"
else
  echo "modules enumeration test failed"
fi
