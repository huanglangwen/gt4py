from gt4py import gtscript
from gt4py import definitions as gt_defs

sd = gtscript.Field[float]
backend_t = "mlir"
#backend_t = "dawn:cxxopt"

# , splitters={gtscript.I: ['b0', 'ie'], gtscript.J: ['j0', 'je'], gtscript.K: ['k0', 'ke']})
@gtscript.stencil(backend=backend_t)
def laplace_index(in_f: sd, out_f: sd):
    with computation(PARALLEL), interval(...):
        out_f = -4.0 * in_f + (
            in_f[-1, 0, 0] + in_f[1, 0, 0] + in_f[0, 1, 0] + in_f[0, -1, 0]
        )

    with computation(PARALLEL), interval(30, None):
        out_f = -out_f
