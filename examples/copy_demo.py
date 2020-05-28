#!/usr/bin/env python
import dawn4py
import gt4py
from gt4py import gtscript
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import traceback as tb

py_backend = 'numpy'
cpu_backend = 'dawn:gtmc'
gpu_backend = 'dawn:cuda'
#gpu_backend = 'dawn:gtcuda'
rebuild = True
managed_mem = False

@gtscript.stencil(backend=py_backend, rebuild=rebuild)
def copy_py(in_f: gtscript.Field[float], out_f: gtscript.Field[float]):
    from __gtscript__ import computation, interval, PARALLEL
    with computation(PARALLEL), interval(...):
        out_f = in_f

@gtscript.stencil(backend=cpu_backend, rebuild=rebuild)
def copy_mc(in_f: gtscript.Field[float], out_f: gtscript.Field[float]):
    from __gtscript__ import computation, interval, PARALLEL
    with computation(PARALLEL), interval(...):
        out_f = in_f

@gtscript.stencil(backend=gpu_backend, rebuild=rebuild)
def copy_cuda(in_f: gtscript.Field[float], out_f: gtscript.Field[float]):
    from __gtscript__ import computation, interval, PARALLEL
    with computation(PARALLEL), interval(...):
        out_f = in_f


def main():
    # Allocate storages
    vsize = 64
    hsize = vsize * 2
    shape = (hsize, hsize, vsize)
    nhalo = 3

    in_data = np.fromfunction(lambda i, j, k: np.sin(i) * np.sin(j), shape, dtype=float)

    in_fld_py = gt4py.storage.from_array(
        data=in_data, backend=py_backend, default_origin=(nhalo, nhalo, 0), shape=shape
    )

    out_fld_py = gt4py.storage.zeros(
        shape=shape, default_origin=(nhalo, nhalo, 0), dtype=float, backend=py_backend
    )

    in_fld_mc = gt4py.storage.from_array(
        data=in_data, backend=cpu_backend, default_origin=(nhalo, nhalo, 0), shape=shape
    )

    out_fld_mc = gt4py.storage.zeros(
        shape=shape, default_origin=(nhalo, nhalo, 0), dtype=float, backend=cpu_backend
    )

    in_fld_cuda = gt4py.storage.from_array(
        data=in_data, backend=gpu_backend, default_origin=(nhalo, nhalo, 0), shape=shape, managed_memory=managed_mem
    )

    out_fld_cuda = gt4py.storage.zeros(
        shape=shape, default_origin=(nhalo, nhalo, 0), dtype=float, backend=gpu_backend, managed_memory=managed_mem
    )    

    t_start = time.perf_counter()
    copy_py(in_fld_py, out_fld_py, domain=(hsize - nhalo * 2, hsize - nhalo * 2, vsize))
    t_stop = time.perf_counter() - t_start
    print("numpy time: %.6lf" % t_stop)
    np.savetxt("copy_py_output.csv", out_fld_py.data.flatten(), delimiter=',')

    t_start = time.perf_counter()
    copy_mc(in_fld_mc, out_fld_mc, domain=(hsize - nhalo * 2, hsize - nhalo * 2, vsize))
    t_stop = time.perf_counter() - t_start
    print("cpu time: %.6lf" % t_stop)
    np.savetxt("copy_cpu_output.csv", out_fld_mc.data.flatten(), delimiter=',')

    t_start = time.perf_counter()
    copy_cuda(in_fld_cuda, out_fld_cuda, domain=(hsize - nhalo * 2, hsize - nhalo * 2, vsize))
    t_stop = time.perf_counter() - t_start
    print("gpu time: %.6lf" % t_stop)
    np.savetxt("copy_gpu_output.csv", np.asarray(out_fld_cuda.data).flatten(), delimiter=',')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e: # Ctrl-C
        print("Closing gracefully on keyboard interrupt...")
    except Exception as e:
        print('ERROR: ' + str(e))
        tb.print_exc()
