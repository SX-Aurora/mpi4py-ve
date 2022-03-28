# ---

import mpi4pyve
try: mpi4pyve.get_include()
except: pass
try: mpi4pyve.get_config()
except: pass

# ---

def test_mpi4pyve_rc():
    import mpi4pyve.rc
    mpi4pyve.rc(
    initialize = True,
    threads = True,
    thread_level = 'multiple',
    finalize = None,
    fast_reduce = True,
    recv_mprobe = True,
    errors = 'exception',
    )
    try: mpi4pyve.rc(qwerty=False)
    except TypeError: pass
    else: raise RuntimeError

test_mpi4pyve_rc()

# ---

def test_mpi4pyve_profile():
    import mpi4pyve
    def mpi4pyve_profile(*args, **kargs):
        try: mpi4pyve.profile(*args, **kargs)
        except ValueError: pass
    import warnings
    warnings.simplefilter('ignore')
    mpi4pyve_profile('mpe')
    mpi4pyve_profile('mpe', path="/usr/lib")
    mpi4pyve_profile('mpe', path=["/usr/lib"])
    mpi4pyve_profile('mpe', logfile="mpi4pyve")
    mpi4pyve_profile('mpe', logfile="mpi4pyve")
    mpi4pyve_profile('vt')
    mpi4pyve_profile('vt', path="/usr/lib")
    mpi4pyve_profile('vt', path=["/usr/lib"])
    mpi4pyve_profile('vt', logfile="mpi4pyve")
    mpi4pyve_profile('vt', logfile="mpi4pyve")
    mpi4pyve_profile('@querty')
    mpi4pyve_profile('c', path=["/usr/lib", "/usr/lib64"])
    mpi4pyve_profile('m', path=["/usr/lib", "/usr/lib64"])
    mpi4pyve_profile('dl', path=["/usr/lib", "/usr/lib64"])
    mpi4pyve_profile('hosts', path=["/etc"])

test_mpi4pyve_profile()

# ---

import mpi4pyve.__main__
import mpi4pyve.bench
import mpi4pyve.futures
import mpi4pyve.futures.__main__
import mpi4pyve.futures.server
import mpi4pyve.run
