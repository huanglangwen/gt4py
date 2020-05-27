import gt4py
from gt4py import gtscript

sd = gtscript.Field[float]


@gtscript.stencil(backend="mlir", rebuild=True)
def UVbKE(uc: sd, vc: sd, cosa: sd, rsina: sd, ub: sd, vb: sd, *, dt5: float):
    # main_ub
    with computation(PARALLEL), interval(...):
        ub = dt5 * (uc[0, -1, 0] + uc - (vc[-1, 0, 0] + vc) * cosa) * rsina

    # main_vb
    with computation(PARALLEL), interval(...):
        vb = dt5 * (vc[-1, 0, 0] + vc - (uc[0, -1, 0] + uc) * cosa) * rsina
