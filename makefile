.PHONY: default
default: build

PYTHON  = python$(py)
MPIEXEC = mpirun-py

PLAT = manylinux1_x86_64
SETPLAT = _PYTHON_HOST_PLATFORM=${PLAT}
MPIOPT = MPICFG=necmpi

# ----

.PHONY: config build test
#config:
#	${PYTHON} setup.py config $(opt)
build:
	#${PYTHON} setup.py build $(opt)
	${SETPLAT} ${MPIOPT} ${PYTHON} -m build
test:
	${VALGRIND} ${PYTHON} ${PWD}/test/runtests.py
test-%:
	${MPIEXEC} -veo -np $* ${VALGRIND} $$(which python) ${PWD}/test/runtests.py
#	${MPIEXEC} -np $* ${VALGRIND} ${PYTHON} ${PWD}/test/runtests.py

#.PHONY: srcbuild srcclean
.PHONY: srcclean
#srcbuild:
#	${PYTHON} setup.py build_src $(opt)
srcclean:

	${RM} src/mpi4pyve/MPI.c
	${RM} src/mpi4pyve/include/mpi4pyve/MPI.h
	${RM} src/mpi4pyve/include/mpi4pyve/MPI_api.h

.PHONY: clean distclean fullclean
clean:
	${RM} -rf build/
	#${PYTHON} setup.py clean --all
distclean: clean
	-${RM} -r build  _configtest* *.py[co]
	-${RM} -r MANIFEST dist mpi4py-ve.egg-info
	-${RM} -r conf/__pycache__ test/__pycache__
	-${RM} -r demo/__pycache__ src/mpi4pyve/__pycache__
	-find conf -name '*.py[co]' -exec rm -f {} ';'
	-find demo -name '*.py[co]' -exec rm -f {} ';'
	-find test -name '*.py[co]' -exec rm -f {} ';'
	-find src  -name '*.py[co]' -exec rm -f {} ';'
fullclean: distclean srcclean docsclean
	-find . -name '*~' -exec rm -f {} ';'

# ----

.PHONY: install uninstall
#install: build
install:
	#${PYTHON} setup.py install --prefix='' --user $(opt)
	${MPIOPT} ${PYTHON} -m pip install .
uninstall:
	#-${RM} -r $(shell ${PYTHON} -m site --user-site)/mpi4py-ve
	#-${RM} -r $(shell ${PYTHON} -m site --user-site)/mpi4py-ve-*-py*.egg-info
	${PYTHON} -m pip uninstall mpi4py-ve

# ----

#.PHONY: docs docs-html docs-pdf docs-misc
.PHONY: docs docs-html
#docs: docs-html docs-pdf docs-misc
docs: docs-html
#docs-html: rst2html sphinx-html epysoc-html
docs-html: rst2html
#docs-pdf:  sphinx-pdf epydoc-pdf
#docs-misc: sphinx-man sphinx-info

RST2HTML = $(shell command -v rst2html || command -v rst2html.py || false)
RST2HTMLOPTS  = --input-encoding=utf-8
RST2HTMLOPTS += --no-compact-lists
RST2HTMLOPTS += --cloak-email-addresses
.PHONY: rst2html
rst2html:
	${RST2HTML} ${RST2HTMLOPTS} ./README.rst  > docs/README.html
	${RST2HTML} ${RST2HTMLOPTS} docs/index.rst > docs/index.html

.PHONY: docsclean
docsclean:
	-${RM} docs/*.info docs/*.[137]
	-${RM} docs/*.html docs/*.pdf
	-${RM} -r docs/usrman docs/apiref

# ----

.PHONY: sdist
sdist: docs
#	${PYTHON} setup.py sdist $(opt)
	${MPIOPT} ${PYTHON} -m build --sdist

# ----
