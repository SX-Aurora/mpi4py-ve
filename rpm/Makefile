.PHONY:mpi4py-ve

all:mpi4py-ve

BASEDIR = ..

# Parameters passed into the spec file
PLAT             = el8
ARCH             = x86_64
VERSION_PYTHON   = python3.6
VERSION_MPI4PYVE = 1.0.0
VERSION_RPM      = 1
RELEASE          = 1

DISTDIR  = $(BASEDIR)/dist
RPMDIR   = $(DISTDIR)/rpm/RPMBUILD
SPECS    = $(RPMDIR)/SPECS
RPMS     = $(RPMDIR)/RPMS
SOURCES  = $(RPMDIR)/SOURCES

# convert 'python3.x -> python3x'
$(eval REQUIRES_PYTHON = $(shell echo $(VERSION_PYTHON) | sed -e s"/\.//"))

PRODUCT        = nec-$(REQUIRES_PYTHON)-mpi4py-ve-$(VERSION_MPI4PYVE)
SRC_MPI4PYVE   = nec-$(REQUIRES_PYTHON)-mpi4py-ve-$(VERSION_MPI4PYVE)-$(VERSION_RPM)-$(RELEASE)
RPM_MPI4PYVE   = $(SRC_MPI4PYVE).$(PLAT).$(ARCH).rpm

# Options of rpmbuild command
$(eval SPECDEF  =--define "_topdir            `cd ..;pwd`")
$(eval SPECDEF +=--define "python_version     $(VERSION_PYTHON)")
$(eval SPECDEF +=--define "mpi4py_ve_version  $(VERSION_MPI4PYVE)")
$(eval SPECDEF +=--define "rpm_version        $(VERSION_RPM)")
$(eval SPECDEF +=--define "mpi4py_ve_platform $(PLAT)")
$(eval SPECDEF +=--define "mpi4py_ve_arch     $(ARCH)")
$(eval SPECDEF +=--define "rpm_release        $(RELEASE)")
$(eval SPECDEF +=--define "product            $(PRODUCT)")

mpi4py-ve: $(RPMS)/$(ARCH)/$(RPM_MPI4PYVE)

# Creates Specfiles
$(SPECS)/$(RPM_MPI4PYVE:%.rpm=%.spec):
	mkdir -p $(SPECS) $(RPMS) $(SOURCES)
	cp -p SPECS/nec-python3-mpi4py-ve $@

# Creates RPM files
$(RPMS)/$(ARCH)/$(RPM_MPI4PYVE): $(SPECS)/$(RPM_MPI4PYVE:%.rpm=%.spec)
	$(eval SRC    = $(SRC_MPI4PYVE))
	$(eval SRCDIR = $(SOURCES)/$(SRC))
	rm -rf $(SRCDIR)
	mkdir -p $(SRCDIR)
	cp -p $(DISTDIR)/mpi4py_ve*.whl $(SRCDIR)/
	cd $(SRCDIR) && unzip mpi4py_ve*.whl && rm -f mpi4py_ve*.whl
	cd $(SOURCES) && tar zcvf $(SRC).tar.gz $(SRC)
	cd $(SPECS) && rpmbuild --bb $(SPECDEF) $(^F)

clean:
	rm -rf $(RPMDIR)
